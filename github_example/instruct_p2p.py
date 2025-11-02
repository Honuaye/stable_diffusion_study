import PIL
from PIL import Image  # Requires Pillow: pip install pillow
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DDPMScheduler
from github_example.tools import load_image, get_images_by_frame_and_camera
import cv2
import os
# python  -m   github_example.instruct_pix2pix_waymo_video      --image_dir   ./data/waymo_data/002/images
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to the points3D.txt file')
    args = parser.parse_args()
    image_dir = args.image_dir
    frame_start, frame_end = 0, 20
    cameras = [0, 1, 2]
    result = get_images_by_frame_and_camera(
        root_dir=image_dir,
        frame_range=(frame_start, frame_end),
        camera_ids=cameras
    )
    print(f"找到 {len(result)} 张符合条件的图像：")
    for path in result:
        print(path)
    model_id = "/data/workspace/yhh/aigc/huggingface/instruct_pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    base_seed = 42
    generator = torch.manual_seed(base_seed)
    target_weather="night"
    prompt = (
        f"change the weather to {target_weather}, keep all vehicles, lane lines, signs unchanged; "
        "stable lighting across frames, no sudden changes in sky or road brightness"
    )
    negative_prompt = "blurred lanes, moving vehicles, flickering light, sudden sky changes"
    prompt = (
        # "turn the scene into a night scene, dark sky with stars, "
        "turn the scene into a evening scene, evening sky with stars, "
        # "turn the scene into a dusk scene, dusk sky with stars, "
        "road and lane lines are clearly visible, vehicles and traffic signs remain unchanged, "
        "headlights and street lights are on"
    )
    negative_prompt = (
        "blurred lane lines, moved vehicles, distorted traffic signs, "
        "changed road shape, missing vehicles, altered lane positions"
    )
    # 黑夜场景： 环境车 尾灯(结果：车灯没控制到，但是生成的黑夜还可以啊)
    if False:
        prompt = (
            "turn the scene into a evening scene, evening sky with stars, "
            "road and lane lines are clearly visible, vehicles and traffic signs remain unchanged in shape, color and position, "
            "headlights and street lights are on, all surrounding vehicles have their taillights illuminated (red lights at the rear), "
            "vehicle taillights are bright but do not distort the original vehicle shape or color"
        )
        negative_prompt = (
            "blurred lane lines, moved vehicles, distorted traffic signs, "
            "changed road shape, missing vehicles, altered lane positions, "
            "vehicles with changed color or shape, unlit vehicle taillights, overly bright taillights distorting vehicles"
        )
    # sunset场景： 生成的场景还可以
    if True:
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
    # 雨天 - 一般般(凑合用)，场景没有变差，路面变湿了，但是没有下雨雨滴的感觉
    # if True:
    if False:
        prompt = (
            "turn the scene into a rainy day, heavy rain falling with visible raindrops, "
            "road surface is wet and reflects car headlights and street lights, "
            "dark gray sky with overcast clouds, "
            "all vehicles, lane lines, road markings, and traffic signs remain unchanged in position and shape, "
            "windshield wipers on vehicles are active, "
            "visibility is slightly reduced but road elements are still clearly recognizable"
        )
        # negative_prompt = "rain, fog, snow, blurry, distorted lane lines, missing vehicles"
        negative_prompt = (
            "snow, fog, night, sunny, blurred lane lines, moved vehicles, "
            "distorted traffic signs, dry road, no raindrops"

            "blurred lane lines, moved vehicles, distorted traffic signs, "
            "changed road shape, missing vehicles, altered lane positions"
        )
    # 雪天 - 没达到效果
    if False:
        prompt = (
            "turn the scene into a snowy day, heavy snowflakes falling densely in the air, "
            "light snow covering the road surface but lane lines still visible through the snow, "
            "vehicles and traffic signs dusted with snow, "
            "cold blue-tinted lighting, overcast gray sky, "
            "all vehicles, lane markings, and traffic signs remain in their original positions and shapes, "
            "snow accumulating on vehicle roofs and the edges of the road, "
            "visibility slightly reduced by falling snow but critical road elements are clear"
        )
        negative_prompt = (
            "rain, fog, sunny, clear sky, no snow, "
            "blurred lane lines, moved vehicles, distorted signs, "
            "heavy snow covering lane lines completely"
        )
    # 沙尘暴天气 -  - 没达到效果
    if False:
    # if True:
        prompt = (
            "turn the scene into a severe sandstorm, dense sand and dust particles swirling densely in the air, "
            "vehicles and traffic signs partially covered with a layer of sand but their outlines remain recognizable, "
            "brownish-yellow haze dominating the atmosphere, reduced visibility with a hazy effect, "
            "road surface covered in a thin layer of sand, "
            "all vehicles maintain their original positions and driving directions, "
            "lane lines are faintly visible beneath the dust, traffic rules remain strictly followed"
        )
        negative_prompt = (
            "clear sky, rain, snow, fog, clean vehicles, "
            "obscured vehicle outlines, missing traffic signs, "
            "vehicles violating traffic rules, completely invisible lane lines"
        )
    output_path="./output/waymo_video/"
    i = 0
    for src_img_path in result:
        print(src_img_path)
        filename = os.path.basename(src_img_path)
        src_image = load_image(src_img_path)
        raw_img_size = src_image.size
        src_image = src_image.resize((512, 512), Image.LANCZOS)
        current_generator = torch.manual_seed(base_seed + i)  # 种子递增，既关联又有差异
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=src_image,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                # image_guidance_scale=1.5,  # 强图像引导，保留结构
                # guidance_scale=8.0,
                # strength=strength,  # 低强度，减少帧间跳变
                # generator=current_generator,
            ).images[0]
            result = result.resize((1920, 1280), Image.LANCZOS)
            # import pdb; pdb.set_trace()
            save_path = os.path.join(output_path, filename)
            result.save(save_path)
        i=i+1
if __name__ == "__main__":
    main()


