import cv2
import numpy as np
import pyttsx3  # 語音合成模組

def detect_circles_in_mask(mask, output_image, color_name, color_bgr):
    """
    偵測遮罩中的圓形並標記在圖片上
    :param mask: 二值遮罩
    :param output_image: 輸出的圖片
    :param color_name: 顏色名稱 (Red, Yellow, Green)
    :param color_bgr: 顯示的顏色 (B, G, R)
    :return: 是否有偵測到圓形
    """
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=20, minRadius=10, maxRadius=100)
    detected = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(output_image, (x, y), r, color_bgr, 2)  # 繪製圓形邊框
            cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)  # 標記圓心
            cv2.putText(output_image, f"{color_name} Light", (x - 30, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            detected = True
    return detected

def main(image_path):
    # 初始化語音引擎
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # 設定語速
    engine.setProperty('volume', 0.9)  # 設定音量

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        engine.say("Image not found!")
        engine.runAndWait()
        return

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅、黃、綠燈的 HSV 範圍
    color_ranges = {
        "Red": [(np.array([0, 100, 80]), np.array([10, 255, 255])),
                (np.array([170, 100, 80]), np.array([180, 255, 255]))],
        "Yellow": [(np.array([15, 100, 80]), np.array([35, 255, 255]))],
        "Green": [(np.array([40, 120, 100]), np.array([90, 255, 255]))]
    }

    # 建立輸出圖片
    output_image = image.copy()

    # 初始化燈光狀態
    detected_light = "No Light Detected"

    # 檢測每個顏色的圓形
    for color_name, ranges in color_ranges.items():
        combined_mask = None
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            if combined_mask is None:
                combined_mask = color_mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # 顯示遮罩
        cv2.imshow(f"{color_name} Mask", combined_mask)

        # 偵測圓形
        detected = detect_circles_in_mask(combined_mask, output_image, color_name,
                                          {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0)}[color_name])
        if detected:
            detected_light = f"{color_name} Light"
            engine.say(f"Detected {color_name} Light")
            engine.runAndWait()
            break  # 若已偵測到燈光，結束迴圈

    # 顯示目前燈光狀態
    cv2.putText(output_image, f"Current Light: {detected_light}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 顯示結果
    cv2.imshow("Traffic Light Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 測試圖片
main("red1.jpg")
