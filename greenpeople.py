import cv2
import numpy as np
import pyttsx3  # 語音合成模組

def detect_green_person(mask, output_image):
    """
    偵測小綠人綠燈
    :param mask: 綠色遮罩
    :param output_image: 輸出的圖片
    :return: 是否有偵測到小綠人綠燈
    """
    # 使用形態學操作改善遮罩品質
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 偵測輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for contour in contours:
        # 計算輪廓的邊界框
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # 顯示所有輪廓（調試用）
        print(f"Contour at ({x}, {y}) with width={w}, height={h}, area={area}")
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 1)  # 藍色框顯示所有輪廓

        # 篩選合理大小的輪廓
        if 10 < w < 300 and 20 < h < 500 and area > 50:  # 調整條件
            aspect_ratio = h / w
            if 1.0 < aspect_ratio < 6.0:  # 調整長寬比
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠色框標記目標
                cv2.putText(output_image, "Green Person Detected", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detected = True

    return detected

def main(image_path):
    # 初始化語音引擎
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # 語速
    engine.setProperty('volume', 0.9)  # 音量

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        engine.say("Image not found!")
        engine.runAndWait()
        return

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義綠燈的 HSV 範圍
    lower_green = np.array([40, 50, 50])  # 放寬範圍
    upper_green = np.array([90, 255, 255])

    # 建立遮罩
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 顯示遮罩
    cv2.imshow("Green Mask", green_mask)

    # 偵測小綠人綠燈
    output_image = image.copy()
    detected = detect_green_person(green_mask, output_image)

    # 語音通知
    if detected:
        engine.say("Green person detected!")
    else:
        engine.say("No green person detected.")
    engine.runAndWait()

    # 顯示結果
    cv2.imshow("Traffic Light Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 測試圖片
main("greenpeople.jpg")
