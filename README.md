# Python 影片風格轉換

這個專案提供兩種影片風格轉換方式：

- 使用 PyTorch 風格轉換模型，將影片轉換成 Candy、Mosaic、Rain Princess 或 Udnie 等風格。
- 使用 `.cube` 3D LUT 對影片套用色彩風格。

## 專案結構

```text
.
├── inference.py          # 使用 PyTorch 模型進行風格轉換
├── inference_lut.py      # 使用 .cube LUT 進行色彩轉換
├── transformer_net.py    # TransformerNet 模型定義
├── candy.pth             # Candy 模型權重
├── mosaic.pth            # Mosaic 模型權重
├── rain_princess.pth     # Rain Princess 模型權重
├── udnie.pth             # Udnie 模型權重
└── lut/                  # .cube LUT 檔案
```

## 環境需求

- Python 3.9 以上
- NVIDIA GPU 與 CUDA 12.8（可選；沒有 GPU 時會使用 CPU）
- 一個可由 OpenCV 讀取的輸入影片，例如 `test.mp4`

## 安裝套件

建議先建立並啟用虛擬環境：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

安裝 PyTorch CUDA 12.8 版本：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安裝其他相依套件：

```powershell
pip install tqdm opencv-python numpy Pillow pillow-lut
```

如果只使用 CPU，也可以改用 PyTorch 官方提供的 CPU 或其他 CUDA 版本安裝指令。

## 使用 PyTorch 風格模型

1. 將輸入影片命名為 `test.mp4`，放在專案根目錄。
2. 開啟 `inference.py`，設定要使用的模型權重：

   ```python
   MODEL_WEIGHT_PATH = "candy.pth"
   ```

   可替換為 `mosaic.pth`、`rain_princess.pth` 或 `udnie.pth`。

3. 執行：

   ```powershell
   python inference.py
   ```

處理完成後會產生 `output_stylized.mp4`。輸入與輸出檔名也可以直接在 `inference.py` 的 `VIDEO_INPUT_PATH` 與 `VIDEO_OUTPUT_PATH` 修改。

## 使用 LUT 色彩轉換

1. 將輸入影片命名為 `test.mp4`，放在專案根目錄。
2. 開啟 `inference_lut.py`，選擇 LUT 檔案：

   ```python
   LUT_FILE = "lut/BlueHour.cube"
   ```

   `lut/` 目錄內已提供多個 `.cube` 檔案可使用。
3. 執行：

   ```powershell
   python inference_lut.py
   ```

處理完成後會產生 `output_lut.mp4`。影片輸入與輸出檔名可在程式中的 `VIDEO_IN` 與 `VIDEO_OUT` 修改。

## 注意事項

- 輸出影片使用 OpenCV 的 `mp4v` 編碼器；若系統缺少對應編碼器，請確認 OpenCV 安裝狀態或改用其他影片編碼設定。
- PyTorch 腳本會自動選擇 CUDA 或 CPU。CUDA 是否可用可透過以下指令確認：

  ```powershell
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- 處理高解析度或長影片時，GPU 記憶體與系統儲存空間需求會增加。