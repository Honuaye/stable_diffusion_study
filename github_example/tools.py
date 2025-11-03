from PIL import Image  # Requires Pillow: pip install pillow
import torch
import os
import re
from typing import List, Tuple
import cv2
from pathlib import Path
from typing import Iterable, Union, Tuple
import argparse

def download_image(path):
    with Image.open(path) as img:
        img = img.convert('RGB')
        return img
    # with Image.open(path).convert('RGB') as img:
    #     # 转为 (H, W, 3) 格式，与 mask 的 (H, W) 扩展为3通道一致
    #     arr = np.array(img, dtype=np.float32)
    #     rgb = torch.tensor(arr)

def imgs_to_video(
    img_paths, out_file, fps: float = 5,
    size: Tuple[int, int] = None,
    fourcc_str: str = 'mp4v') -> None:

    first = cv2.imread(str(img_paths[0]))
    h_out, w_out = first.shape[:2]
    if size is not None:
        w_out, h_out = size

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(str(out_file), fourcc, fps, (w_out, h_out))
    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if frame.shape[:2] != (h_out, w_out):
            frame = cv2.resize(frame, (w_out, h_out))
        vw.write(frame)
    vw.release()

def get_images_by_frame_and_camera(
    root_dir: str,
    frame_range: Tuple[int, int],
    camera_ids: List[int]
) -> List[str]:
    pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
    matched_files = []
    for filename in os.listdir(root_dir):
        match = pattern.match(filename)
        if not match:
            continue
        frame_id = int(match.group(1))
        camera_id = int(match.group(2))
        if (frame_range[0] <= frame_id <= frame_range[1]) and (camera_id in camera_ids):
            file_path = os.path.abspath(os.path.join(root_dir, filename))
            # import pdb; pdb.set_trace()
            matched_files.append((frame_id, camera_id, file_path))
    matched_files.sort(key=lambda x: (x[0], x[1]))
    return [fp for (fid, cid, fp) in matched_files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to the points3D.txt file')
    args = parser.parse_args()
    image_dir = args.image_dir
    frame_start, frame_end = 0, 20
    frame_start, frame_end = 0, 198
    cameras = [0]
    image_paths = get_images_by_frame_and_camera(
        root_dir=image_dir,
        frame_range=(frame_start, frame_end),
        camera_ids=cameras
    )
    print(f"找到 {len(result)} 张符合条件的图像：")
    for path in image_paths:
        print(path)
    # import pdb; pdb.set_trace()
    video_output_path = os.path.join(image_dir, "video.mp4")
    imgs_to_video(image_paths, video_output_path, fps=5, size={960, 640})

if __name__ == '__main__':
    main()


