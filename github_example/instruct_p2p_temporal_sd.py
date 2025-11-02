"""
temporal_instruct_smoothing.py

用途：
- 输入：一组 GT 帧（frame_000.png ... frame_009.png）和对应的 Instruct prompt
- 先用 InstructPix2Pix 对每帧做风格迁移（或假设你已有 pix2pix_outputs）
- 然后基于 diffusers 对迁移结果做潜空间光流对齐 + 带初始化的 image2image 采样
- 输出：smoothed stylized frames

说明：
- 代码用光流 (Farneback) 对齐像素坐标，然后将像素坐标映射到 VAE latent 尺度做 remap。
- 若你的 pipeline 暴露 VAE encoder/decoder（diffusers pipeline 通常有），我们用它来计算 latents。
- 若使用 InstructPix2PixPipeline 名称不一致，可把 pipeline 换为 StableDiffusionImg2ImgPipeline 并调整 prompt。
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# diffusers imports
from diffusers import InstructPix2PixPipeline  # 若报错，改用 StableDiffusionImg2ImgPipeline
# from diffusers import StableDiffusionImg2ImgPipeline as Img2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- 用户配置 ----------
input_gt_dir = "./gt_frames"                # 原始 GT 帧（用于生成 pseudo-style 或用于配准）
pix2pix_outputs_dir = "./pix2pix_outputs"   # 如果你已用 InstructPix2Pix 得到的直接放这里（可选）
output_dir = "./temporal_diffusers_out"
os.makedirs(output_dir, exist_ok=True)

# 模型与 prompt
model_id = "timbrooks/instruct-pix2pix"  # 例子模型 id；本地或 huggingface 模型均可
prompt = "Make the scene look snowy, with gently falling snow and soft lighting."
strength = 0.6   # img2img strength（0-1），越低越保留 init latents（或 init image）
alpha_latent_mix = 0.85  # 用于混合 warped_prev_latent 与 current_encoded_latent 的权重（靠近1更保留前帧）
num_inference_steps = 20
guidance_scale = 7.5

# optical flow params
fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# transform
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# --------- 加载 pipeline ----------
pipe = InstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
pipe = pipe.to(device)
pipe.scheduler = pipe.scheduler  # 可替换 scheduler 以加速/质量 tradeoff

# 获取 VAE 组件（不同模型命名可能不同）
vae = pipe.vae
unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

# helper: read and normalize image
def load_img(path):
    img = Image.open(path).convert("RGB")
    return img

def pil_to_numpy(img: Image.Image):
    return np.array(img).astype(np.float32) / 255.0

def numpy_to_pil(arr):
    arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(arr)

# encode image to latent (using pipeline's VAE)
def encode_image_to_latent(img_pil, device=device):
    # expects PIL image in [0,1] scaled -> convert to tensor batch
    image = to_tensor(img_pil).unsqueeze(0).to(device)
    # scale to [-1,1]
    image = 2.0 * image - 1.0
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()  # shape [1, C, H/8, W/8] depending on model
        latents = latents * 0.18215  # typical scaling (may differ by VAE impl)
    return latents  # torch tensor

# decode latent to image
def decode_latent_to_image(latents, device=device):
    with torch.no_grad():
        latents = latents / 0.18215
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0,1)
    pil = to_pil(image.squeeze(0).cpu())
    return pil

# warp a float32 image (H,W,3) by optical flow (u,v) where flow given from prev->curr
def warp_image_by_flow(image, flow):
    h,w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)
    warped = cv2.remap((image*255).astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.float32) / 255.0

# warp latent by flow: upsample flow to latent resolution and remap each channel
def warp_latent_by_flow(latent: torch.Tensor, flow_np):
    """
    latent: torch [1, C, H_lat, W_lat]
    flow_np: numpy flow at image resolution (H_img, W_img, 2)
    """
    _, C, H_lat, W_lat = latent.shape
    # resize flow to latent resolution
    flow_small = cv2.resize(flow_np, (W_lat, H_lat), interpolation=cv2.INTER_LINEAR)
    # scale flow by factor of latent/image ratio
    h_img, w_img = flow_np.shape[:2]
    scale_x = W_lat / float(w_img)
    scale_y = H_lat / float(h_img)
    flow_small[...,0] *= scale_x
    flow_small[...,1] *= scale_y

    # create sampling grid (normalized -1..1) expected by grid_sample
    grid_x, grid_y = np.meshgrid(np.linspace(-1,1,W_lat), np.linspace(-1,1,H_lat))
    # apply offset in normalized coordinates
    # convert flow_small from pixels to normalized coords
    flow_norm_x = flow_small[...,0] / (W_lat/2.0)
    flow_norm_y = flow_small[...,1] / (H_lat/2.0)
    grid = np.stack([grid_x + flow_norm_x, grid_y + flow_norm_y], axis=-1)
    grid_t = torch.from_numpy(grid).unsqueeze(0).to(latent.device).to(latent.dtype)
    # latent is [1,C,H,W], grid_sample expects [N,H,W,2]
    warped = torch.nn.functional.grid_sample(latent, grid_t, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

# process frames
frame_files = sorted([os.path.join(input_gt_dir, f) for f in os.listdir(input_gt_dir) if f.lower().endswith((".png",".jpg"))])
n = len(frame_files)
assert n>0, "No frames found"

prev_stylized = None
prev_latent = None

for i, gt_path in enumerate(frame_files):
    print(f"Processing frame {i+1}/{n}: {gt_path}")
    gt_img = load_img(gt_path)
    gt_np = pil_to_numpy(gt_img)

    # Step A: if you already have a pix2pix output for this GT, load it; else run InstructPix2Pix once to get a pseudo-target (optional)
    pix2pix_out_path = os.path.join(pix2pix_outputs_dir, os.path.basename(gt_path))
    if os.path.exists(pix2pix_out_path):
        target_stylized_pil = load_img(pix2pix_out_path)
    else:
        # run one-shot InstructPix2Pix to get initial stylized target (this also creates initial stylized frames)
        out = pipe(prompt=prompt, image=gt_img, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        target_stylized_pil = out.images[0]

    # encode current GT to latent (this latent represents content of GT)
    curr_content_latent = encode_image_to_latent(gt_img)

    if i == 0:
        # first frame: just produce stylized result from GT
        # we already have target_stylized_pil from pix2pix or from pipe above
        stylized = target_stylized_pil
        # store stylized latent
        prev_stylized = stylized
        prev_latent = encode_image_to_latent(stylized)
        # save
        stylized.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
        continue

    # For i>0: compute optical flow from previous GT to current GT (or prev_stylized->curr_gt)
    # Option A: flow from prev_stylized to current GT gives better alignment of style -> use prev_stylized & current GT
    prev_img_np = pil_to_numpy(prev_stylized)
    curr_img_np = pil_to_numpy(gt_img)

    prev_gray = (cv2.cvtColor((prev_img_np*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)).astype(np.uint8)
    curr_gray = (cv2.cvtColor((curr_img_np*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **fb_params)  # flow from prev->curr pixels

    # warp prev_latent to current view
    warped_prev_latent = warp_latent_by_flow(prev_latent, flow)  # torch tensor

    # mix warped prev latent with current content latent to produce initialization
    init_latent = alpha_latent_mix * warped_prev_latent + (1.0 - alpha_latent_mix) * curr_content_latent.to(warped_prev_latent.dtype)

    # Now run image2image sampling with `init_latents` set to init_latent (diffusers allows passing latents to pipeline)
    # NOTE: API: InstructPix2PixPipeline accepts "image" param; some pipelines accept "latents" via low-level denoising.
    # We'll use pipe with "image" parameter but initialize by decoding init_latent to image as init_image (practical and compatible).
    init_img_pil = decode_latent_to_image(init_latent)

    # Run one more guided pass to follow prompt but keep init appearance
    out = pipe(prompt=prompt, image=init_img_pil, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength)
    stylized = out.images[0]

    # save stylized
    stylized.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

    # update prev variables
    prev_stylized = stylized
    prev_latent = encode_image_to_latent(stylized)

print("Done. Smoothed stylized frames saved to:", output_dir)
