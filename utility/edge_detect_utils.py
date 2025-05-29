from PIL import Image
import numpy as np
import cv2
import sys
sys.path.append("./utility")
import Canny.canny_utils as canny_utils
import Canny.canny_edge_detector as ced

def edge_detector(input_image):
    input_image = input_image.resize((512, 512), Image.LANCZOS)
    imgs = canny_utils.process_pil_image(input_image)
    detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    edges = detector.detect()
    # 转换数据类型并调整维度
    edges = edges[0].astype(np.uint8)  # 确保输入是 uint8
    # 处理单通道图像
    image = edges.squeeze()  # 去除可能的单维度，形状 (H, W)
    image = np.stack([image]*3, axis=-1)  # 直接复制为 (H, W, 3)
    # 转置维度（若需要）
    # image = np.transpose(image, (1, 2, 0))
    # 保存图像
    canny_image = Image.fromarray(image)
    # canny_image.save("output_e2.jpg")
    
    return canny_image
    