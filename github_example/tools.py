import os
import re
from typing import List, Tuple
from PIL import Image

def load_image(src_img_path):
    try:
        return Image.open(src_img_path)
    except IOError:
        print(f"无法打开图片文件: {src_img_path}")
        return None

def get_images_by_frame_and_camera(
    root_dir: str,
    frame_range: Tuple[int, int],
    camera_ids: List[int]
) -> List[str]:
    """
    根据frame id范围和camera id筛选图像路径
    
    参数:
        root_dir: 图像文件夹路径
        frame_range: 帧id范围 (start_frame, end_frame)，包含边界
        camera_ids: 需要筛选的相机id列表
    
    返回:
        符合条件的图像绝对路径列表（按frame id升序排序）
    """
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
            matched_files.append((frame_id, camera_id, file_path))
    matched_files.sort(key=lambda x: (x[0], x[1]))
    return [fp for (fid, cid, fp) in matched_files]


