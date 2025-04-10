import cv2
import numpy as np

# 定義顏色範圍 (HSV)
Blue_lower = np.array([102, 201, 73])
Blue_upper = np.array([220, 255, 255])
Green_lower = np.array([66, 54, 57])
Green_upper = np.array([97, 255, 137])
Yellow_lower = np.array([9, 66, 29])
Yellow_upper = np.array([38, 255, 255])
Red_lower1 = np.array([0, 102, 21])
Red_upper1 = np.array([7, 255, 255])

# 影像參數
Video_num = 1  # 攝影機編號

# 載入遮罩
mask = cv2.imread("mask.png")
capture = cv2.VideoCapture(Video_num)

while True:
    # 讀取影像
    ret, cap_input = capture.read()
    if not ret:
        print("無法讀取影像")
        break
    
    # 應用預載入的遮罩
    after_mask = cv2.bitwise_and(cap_input, mask)
    
    # 轉換為 HSV 色彩空間進行顏色檢測
    hsv = cv2.cvtColor(after_mask, cv2.COLOR_BGR2HSV)
    
    # 生成各顏色的遮罩
    Blue_mask = cv2.inRange(hsv, Blue_lower, Blue_upper)
    Green_mask = cv2.inRange(hsv, Green_lower, Green_upper)
    Yellow_mask = cv2.inRange(hsv, Yellow_lower, Yellow_upper)
    Red_mask = cv2.inRange(hsv, Red_lower1, Red_upper1)
    
    # 計算各顏色區域的非零像素數量
    c1 = cv2.countNonZero(Blue_mask)
    c2 = cv2.countNonZero(Yellow_mask)
    c3 = cv2.countNonZero(Green_mask)
    c4 = cv2.countNonZero(Red_mask)
    
    # 判斷主要顏色
    if (c1 > c2) and (c1 > c4) and (c1 > c3):
        color_state = "Blue"
    elif (c2 > c1) and (c2 > c4) and (c2 > c3):
        color_state = "Yellow"
    elif (c3 > c1) and (c3 > c4) and (c3 > c2):
        color_state = "Green"
    elif (c4 > c1) and (c4 > c2) and (c4 > c3):
        color_state = "Red"
    else:
        color_state = "None"
    
    # 顯示結果
    print(f"🎨 主要顏色: {color_state}")
    cv2.imshow("Masked Image with Color Detection", after_mask)
    
    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 釋放資源
capture.release()
cv2.destroyAllWindows()