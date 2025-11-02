import cv2
import numpy as np
import os

def temporal_smooth_with_flow(input_dir, output_dir, alpha=0.7):
    """
    对 InstructPix2Pix 迁移结果序列进行时序平滑。
    - input_dir: 包含迁移后帧的文件夹路径
    - output_dir: 输出平滑后帧的路径
    - alpha: 当前帧与前一帧的融合系数（0.6~0.8 推荐）

    假设输入图像命名为 frame_000.png ~ frame_009.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有图像路径并排序
    image_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    # 读取首帧
    prev_img = cv2.imread(image_files[0]).astype(np.float32) / 255.0
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_files[0])), 
                (prev_img * 255).astype(np.uint8))

    for i in range(1, len(image_files)):
        curr_img = cv2.imread(image_files[i]).astype(np.float32) / 255.0

        # 计算光流（灰度图）
        prev_gray = cv2.cvtColor((prev_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor((curr_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )

        # 对齐前一帧到当前帧坐标
        h, w = flow.shape[:2]
        flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
        warped_prev = cv2.remap(prev_img, 
                                flow_map[...,0] + flow[...,0],
                                flow_map[...,1] + flow[...,1],
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

        # EMA 平滑
        smoothed = alpha * curr_img + (1 - alpha) * warped_prev

        # 保存
        out_path = os.path.join(output_dir, os.path.basename(image_files[i]))
        cv2.imwrite(out_path, (np.clip(smoothed, 0, 1) * 255).astype(np.uint8))

        # 更新上一帧
        prev_img = smoothed.copy()

        print(f"[{i}/{len(image_files)}] smoothed: {out_path}")

    print("✅ Temporal smoothing done! Output saved to:", output_dir)
