import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformer_net import TransformerNet

# 1. 檔案與硬體配置
VIDEO_INPUT_PATH = "test.mp4"       # 輸入影片檔名
VIDEO_OUTPUT_PATH = "output_stylized.mp4" # 輸出影片檔名
MODEL_WEIGHT_PATH = "candy.pth"          # 權重檔名

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet().to(device)

# 2. 載入官方權重並過濾鍵值
state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=device)
for k in list(state_dict.keys()):
    if "running_mean" in k or "running_var" in k:
        del state_dict[k]
model.load_state_dict(state_dict)
model.eval()

# 啟用報告建議的記憶體格式優化 (NHWC)
model = model.to(memory_format=torch.channels_last)

# 3. 初始化影片讀取與寫入器
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"無法開啟影片檔案：{VIDEO_INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 使用 mp4v 編碼器輸出 MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

print(f"開始處理影片：{VIDEO_INPUT_PATH} ({width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 幀)")

# 4. 逐幀推論並顯示進度條
with torch.no_grad():
    with tqdm(total=total_frames, desc="風格轉換中", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # 前處理：BGR -> RGB -> Tensor (0~255 Float)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
            frame_tensor = frame_tensor.to(memory_format=torch.channels_last)

            # 極速推論 (FP16 autocast)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output_tensor = model(frame_tensor)

            # 後處理：Tensor -> BGR 影像
            output_img = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            out.write(output_bgr)
            pbar.update(1)

# 5. 釋放資源
cap.release()
out.release()
print(f"處理完成！已輸出至：{VIDEO_OUTPUT_PATH}")