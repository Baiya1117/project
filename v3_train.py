# 安裝 YOLOv8 相關套件
# pip uninstall pip setuptools
# pip3 install --upgrade pip
# pip3 install --upgrade setuptools
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics
# pip install ultralytics

import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 載入現有的模型權重
    model = YOLO("models/best_cube_1.pt")

    # 繼續訓練
    model.train(
    data="C:\\Users\\jimmy\\Desktop\\project\\Cube_Color.v3i.yolov11\\data.yaml",  # 數據集配置文件路徑
    epochs=50,                                # 訓練輪數，可以根據需要調整
    imgsz=256,                                # 圖片輸入尺寸，通常為 640x640
    patience=10,  # 等待世代數，無改善就提前停止訓練
    batch=2,                                 # 批次大小，根據你的 GPU 記憶體調整
    name="cube_training2"                     # 訓練結果的保存名稱
)
