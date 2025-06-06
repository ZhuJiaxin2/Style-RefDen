import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name = 'faces_imgs'):    
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs

def process_pil_image(input_image):
    imgs = []
    gray_image = input_image.convert('L')  # 加权公式: 0.299*R + 0.587*G + 0.114*B [[1]](#__1) [[8]](#__8)
    gray_array = np.array(gray_image, dtype=np.float32) / 255.0 
    imgs.append(gray_array)
    return imgs

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

    