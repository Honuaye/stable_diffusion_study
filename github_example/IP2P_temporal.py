import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# diffusers imports
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# from diffusers import InstructPix2PixPipeline  # 若报错，改用 StableDiffusionImg2ImgPipeline
# from diffusers import StableDiffusionImg2ImgPipeline as Img2ImgPipeline
from github_example.tools import download_image, get_images_by_frame_and_camera
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型与 prompt
model_id = "/data/workspace/yhh/aigc/huggingface/instruct_pix2pix"
# model_id = "timbrooks/instruct-pix2pix"
# prompt = "Make the scene look snowy, with gently falling snow and soft lighting."
# prompt = "turn the scene into a evening scene, with soft lighting."
# prompt = "turn the scene into a raining. Water accumulation on the road surface."
# prompt = "Make the scene look snowy"  # 效果没达到雪天效果

prompt = "turn the scene into a evening scene, with soft lighting."
guidance_scale = 7.5

# 傍晚场景 - 效果还行.
if True:
    prompt = "turn the scene into a evening scene, with soft lighting."
    guidance_scale = 7.5


# 沙尘暴场景 - 效果还行. 搭配 guidance_scale = 0.1； （camera4 图像变成全黑）
if False:
# if True:
    # prompt = "Transform the scene into a sandstorm climate"    # 效果非常差
    # 效果还行. 搭配 guidance_scale = 0.1
    prompt = (
        "turn the scene into a severe sandstorm, dense sand and dust particles swirling densely in the air, "
        "vehicles and traffic signs partially covered with a layer of sand but their outlines remain recognizable, "
        "brownish-yellow haze dominating the atmosphere, reduced visibility with a hazy effect, "
        "road surface covered in a thin layer of sand, "
        "all vehicles maintain their original positions and driving directions, "
        "lane lines are faintly visible beneath the dust, traffic rules remain strictly followed"
    )
    guidance_scale = 0.1


# sunset场景： 生成的场景感觉还可以更好
if False:
# if True:
    prompt = (
        "turn the scene into a sunset scene, orange and pink sky with a visible sun (low on the horizon), "
        # "Can see the sunset ahead "
        "warm golden light casting long shadows on the road, "
        "road and lane lines are clearly visible, vehicles and traffic signs remain unchanged in shape, color and position, "
        "vehicle headlights are off (natural sunset light is sufficient), street lights are not yet on"
    )
    negative_prompt = (
        "blurred lane lines, moved vehicles, distorted traffic signs, "
        "changed road shape, missing vehicles, altered lane positions, "
        "night sky, stars, bright sunlight (not sunset), headlights or street lights on, no visible sun"
    )

# 雨天 - 一般般
if True:
# if False:
    prompt = (
        "turn the scene into a rainy day, heavy rain falling with visible raindrops, "
        "road surface is wet and reflects car headlights and street lights, "
        "dark gray sky with overcast clouds, "
        "all vehicles, lane lines, road markings, and traffic signs remain unchanged in position and shape, "
        "windshield wipers on vehicles are active, "
        "visibility is slightly reduced but road elements are still clearly recognizable"
    )
    # 这条 prompt 的效果很好啊！！ 在 fisheye 上;  guidance_scale = 7.5
    # 在 LC数据表现还行 上;  guidance_scale = 10
    prompt = (
        "turn the scene into a rainy day, heavy rain falling with visible raindrops, "
    )
    guidance_scale = 10
    # guidance_scale = 7.5
    # guidance_scale = 0.1
    # guidance_scale = 1
    # guidance_scale = 4
    # guidance_scale = 6
    # guidance_scale = 6.5
    # guidance_scale = 7
    # guidance_scale = 150

strength = 0.6   # img2img strength（0-1），越低越保留 init latents（或 init image）
alpha_latent_mix = 0.85  # 用于混合 warped_prev_latent 与 current_encoded_latent 的权重（靠近1更保留前帧）
num_inference_steps = 20
# optical flow params
fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
to_pil = transforms.ToPILImage() # transform
to_tensor = transforms.ToTensor()
# --------- 加载 pipeline ----------
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
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

def encode_image_to_latent(img_pil, device=device):
    image = to_tensor(img_pil).unsqueeze(0).to(device)
    image = 2.0 * image - 1.0
    image = image.half()  # Cast input to float16 to match model's dtype
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
    return latents

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='image_dir')
    parser.add_argument('--single_frame', type=str, help='single_frame path')
    args = parser.parse_args()
    # image_dir = "/data/workspace/yhh/aigc/aigc_study/data/waymo_data/002/images"
    if args.image_dir is not None:
        image_dir = args.image_dir
        frame_start, frame_end = 0, 10
        # frame_start, frame_end = 0, 198
        # cameras = [0,1,2,3,4]
        # cameras = [7,8,9,10]
        # cameras = [0]
        # cameras = [0,1,2]
        # cameras = [0,3,4]
        cameras = [1,2,5]
        frame_files = get_images_by_frame_and_camera(
            root_dir=image_dir,
            frame_range=(frame_start, frame_end),
            camera_ids=cameras
        )
        for path in frame_files:
            print(path)
        print(f"找到 {len(frame_files)} 张符合条件的图像：")
    else:
        single_frame = args.single_frame
        frame_files = []
        frame_files.append(single_frame)


    # --------- 用户配置 ----------
    pix2pix_outputs_dir = "./output/IP2P_temporal_res/pix2pix_outputs"   # 如果你已用 InstructPix2Pix 得到的直接放这里（可选）
    output_dir = "./output/IP2P_temporal_res/temporal_diffusers_out"
    os.makedirs(pix2pix_outputs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    prev_stylized = None
    prev_latent = None
    only_IP2P = True
    for i, gt_path in enumerate(frame_files):
        print(f"Processing frame {i+1}/{len(frame_files)}: {gt_path}")
        gt_img = load_img(gt_path)
        gt_np = pil_to_numpy(gt_img)
        # Step A: if you already have a pix2pix output for this GT, load it; else run InstructPix2Pix once to get a pseudo-target (optional)
        pix2pix_out_path = os.path.join(pix2pix_outputs_dir, os.path.basename(gt_path))
        # if os.path.exists(pix2pix_out_path):
        #     target_stylized_pil = load_img(pix2pix_out_path)
        # else:
        if True:
        # while(1):
            # import pdb; pdb.set_trace()
            target_stylized_pil = pipe(prompt=prompt, image=gt_img, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
            target_stylized_pil.save(os.path.join(pix2pix_outputs_dir, os.path.basename(gt_path)))
            print('i = ', i)
            if only_IP2P: continue
        print('finish-i = ', i)
        import pdb; pdb.set_trace()

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

if __name__ == "__main__":
    main()




