import cv2
import numpy as np
from PIL import Image
from pillow_lut import load_cube_file
from tqdm import tqdm

VIDEO_IN = "test.mp4"
VIDEO_OUT = "output_lut.mp4"
LUT_FILE = "lut/BlueHour.cube"  # 你的 .cube LUT 檔案

# 1. 載入 LUT
lut = load_cube_file(LUT_FILE)

# 2. 初始化影片
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# 3. 逐幀套用 LUT
with tqdm(total=total_frames, desc="LUT 色彩套用中") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR (OpenCV) -> RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 核心：套用 3D LUT 濾鏡
        lut_img = pil_img.filter(lut)
        
        # 轉回 BGR 寫出
        out_frame = cv2.cvtColor(np.array(lut_img), cv2.COLOR_RGB2BGR)
        out.write(out_frame)
        pbar.update(1)

cap.release()
out.release()
print("LUT 轉換完成！")