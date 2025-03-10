#安裝必要套件
# pip install ultralytics yt-dlp opencv-python opencv-python-headless numpy cvzone

import cv2
import cvzone
import subprocess
import time
from ultralytics import YOLO

# 設定 YouTube 直播網址
youtube_url = "https://www.youtube.com/watch?v=2lVMTs_Fj0M&ab_channel=%E5%8F%B0%E7%81%A3%E5%8D%B3%E6%99%82%E5%BD%B1%E5%83%8F%E7%9B%A3%E8%A6%96%E5%99%A8"

# 使用 yt-dlp 抓取直播串流 URL
def get_stream_url(youtube_url):
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4]", "-g", youtube_url],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()  # 取得串流 URL
    except subprocess.CalledProcessError:
        print("無法獲取 YouTube 串流，請確認 URL 是否正確！")
        return None

stream_url = get_stream_url(youtube_url)
if not stream_url:
    raise Exception("無法獲取串流 請確認 YouTube 直播網址！")

print(" 成功獲取直播串流 URL開始偵測...")

# 讀取直播串流
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise Exception("無法開啟直播串流！")

# 載入 YOLOv8 模型（可換 `yolov8s.pt`）
model = YOLO('yolov8m.pt')

# 只關注 "person" 類別
person_class_id = 0  

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("串流中斷，重新嘗試...")
        time.sleep(5)  # 等待 5 秒再嘗試
        cap = cv2.VideoCapture(stream_url)
        continue

    # YOLO 進行偵測
    results = model(img, stream=True, conf=0.5)
    
    person_count = 0  # 計算行人數量
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 取得邊界框座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # 取得類別與信心度
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]), 2)

            # 只偵測 "person"
            if cls == person_class_id:
                person_count += 1
                
                # 繪製標記框
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
                cvzone.putTextRect(
                    img, f'Person {conf}', (x1, max(35, y1 - 10)), scale=1, thickness=2, offset=5
                )
    
    # 顯示行人計數
    cv2.putText(img, f'Total Persons: {person_count}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示畫面
    cv2.imshow("YouTube Live Pedestrian Detection", img)  

    # 按 `q` 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
