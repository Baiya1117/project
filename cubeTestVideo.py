import cv2
from ultralytics import YOLO

# 加載預訓練的 YOLO 模型
model = YOLO("./models/best_cube_1.pt")  # 換成你使用的模型

# 開啟攝影機或影片文件
cap = cv2.VideoCapture(0)  # 0 為開啟攝影機，或替換為影片檔案路徑
# cap = cv2.VideoCapture(0)  # 0 為開啟攝影機，或替換為影片檔案路徑
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, [0, 0], fx = 1, fy = 1)

    # 使用 YOLO 模型對影像進行偵測
    results = model(frame)

    # 從結果中提取標註過的影像
    annotated_frame = results[0].plot()  # 返回標註過的影像

    # 顯示影像
    cv2.imshow("Car_detect", annotated_frame)

    # 按 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
