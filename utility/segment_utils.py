import cv2
import numpy as np
from PIL import Image
from .MODNet.modnet_inference import person_segment

def portrait_segment(image):
    matte = person_segment(image)
    
    mask = np.array(matte)
    # 二值化处理并提取最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        exit(1)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    target_ratio = 1 / 1
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_h = int(w / target_ratio)
        y = max(0, y - (new_h - h) // 2)  # 垂直居中
        h = new_h
    else:
        new_w = int(h * target_ratio)
        x = max(0, x - (new_w - w) // 2)  # 水平居中
        w = new_w

    image_np = np.array(image)
    cropped = image_np[y:y+h, x:x+w]
    
    cv2.imwrite("output.jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    
    return Image.fromarray(cropped)
    