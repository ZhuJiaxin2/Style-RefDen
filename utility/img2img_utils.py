import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from PIL import Image
import torchvision.transforms as transforms
# style fidelity of ref_uncond_xt. If style_fidelity=1.0, control more important, elif style_fidelity=0.0, prompt more important, else balanced.
# The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original unet. If multiple ControlNets are specified in init, you can set the corresponding scale as a list.

def img2img_with_reference(
    pipe,
    init_image: Image.Image,
    ref_image: Image.Image,
    canny_image: Image.Image,
    prompt: str,
    strength: float = 0.8,
    num_inference_steps: int = 30,
    reference_attn: bool = True,
    reference_adain: bool = False,
    style_fid = 0.5,
    ref_color_fid = 0.85,
    **kwargs
):
    width = kwargs.get('width', 648)
    height = kwargs.get('height', 648)
    
    init_image = init_image.resize((width, height), Image.LANCZOS)
    init_image = np.array(init_image).astype(np.float32) / 255.0
    init_image = (init_image - 0.5) / 0.5
    init_image = torch.from_numpy(init_image.transpose(2,0,1)).unsqueeze(0)

    ref_image = ref_image.resize((width, height), Image.LANCZOS)
    ref_image = np.array(ref_image).astype(np.float32) / 255.0
    ref_image = (ref_image - 0.5) / 0.5
    ref_image = torch.from_numpy(ref_image.transpose(2,0,1)).unsqueeze(0)
    
    canny_image = canny_image.resize((width, height), Image.LANCZOS)
    
    original_dtype = pipe.vae.dtype
    device = pipe.device
    
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    pipe.unet = pipe.unet.to(dtype=torch.float32)
    pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=torch.float32)
    
    with torch.no_grad():
        init_image = init_image.to(device, dtype=torch.float32)
        init_latents = pipe.vae.encode(init_image).latent_dist.sample()
        init_latents = pipe.vae.config.scaling_factor * init_latents
        init_latents_origin = init_latents
        # save_cuda_tensor_as_pil(init_latents_origin, 'init1.jpg')
        # ZJX
        ref_image = ref_image.to(device, dtype=torch.float32)
        ref_latents = pipe.vae.encode(ref_image).latent_dist.sample()
        ref_latents = pipe.vae.config.scaling_factor * ref_latents
        # save_cuda_tensor_as_pil(ref_latents, 'ref1.jpg')
        
        init_latents = init_latents_adain(init_latents, ref_latents)
        
        init_latents = ref_color_fid*init_latents + (1 - ref_color_fid)*init_latents_origin
        # save_cuda_tensor_as_pil(init_latents, 'latent2.jpg')
        
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_idx = max(0, int((1 - strength) * num_inference_steps))
    timesteps = pipe.scheduler.timesteps[start_idx:]
    
    noise = torch.randn_like(init_latents)
    current_timestep = timesteps[0].reshape(1)  
    noisy_latents = pipe.scheduler.add_noise(init_latents, noise, current_timestep)
     
    with torch.autocast(device_type="cuda", dtype=torch.float32):  # 强制使用float32
        result = pipe(
                    prompt=prompt,
                    image=canny_image,
                    ref_image=ref_image,
                    latents=noisy_latents.to(dtype=torch.float32),
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    reference_attn=reference_attn,
                    reference_adain=reference_adain,
                    controlnet_conditioning_scale=1.1,
                    style_fidelity=style_fid, 
                    **kwargs
                )
        
    pipe.vae = pipe.vae.to(dtype=original_dtype)
    pipe.unet = pipe.unet.to(dtype=original_dtype)
    pipe.text_encoder = pipe.text_encoder.to(dtype=original_dtype)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=original_dtype)
    return result


def get_canny_img(input_image):
    image = cv2.Canny(np.array(input_image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def get_canny_512(input_image):
    image = cv2.resize(np.array(input_image), (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    image = cv2.Canny(blurred, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def asian_canny_512(input_image):
    image = cv2.resize(np.array(input_image), (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    image = cv2.Canny(blurred, 70, 150)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def asian_get_enhanced_edges(input_image):
    # 转换为numpy数组
    img = np.array(input_image)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 图像增强 - 对比度拉伸
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))  # 增加clipLimit，减小tileGridSize
    gray = clahe.apply(gray)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Canny边缘检测
    canny = cv2.Canny(blurred, 30, 200)
    # # 中值滤波去除细小噪声
    # edges = cv2.medianBlur(canny, 5)
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel * 255 / np.max(sobel))
    
    
    # 合并边缘检测结果
    edges = cv2.bitwise_or(canny, sobel)
    edges = cv2.bitwise_or(edges, thresh)
    # # 中值滤波去除细小噪声
    edges = cv2.medianBlur(edges, 7)
    
    # edges = cv2.bitwise_or(edges, thresh)
    # edges = cv2.bitwise_or(canny, thresh)
    
    # 形态学操作增强边缘
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 转换为三通道图像
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def europe_get_enhanced_edges(input_image):
    # 转换为numpy数组
    img = np.array(input_image)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 图像增强 - 对比度拉伸
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # # 自适应阈值处理
    # thresh = cv2.adaptiveThreshold(
    #     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY_INV, 11, 2
    # )
    
    # Canny边缘检测
    canny = cv2.Canny(blurred, 30, 200)
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel * 255 / np.max(sobel))
    
    
    # 合并边缘检测结果
    edges = cv2.bitwise_or(canny, sobel)
    # 中值滤波去除细小噪声
    # edges = cv2.medianBlur(edges, 1)
    
    # edges = cv2.bitwise_or(edges, thresh)
    # edges = cv2.bitwise_or(canny, thresh)
    
#     # 形态学操作增强边缘
#     kernel = np.ones((3,3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)
#     edges = cv2.erode(edges, kernel, iterations=1)
    
    # 转换为三通道图像
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def init_image_adain(input_image, ref_image):
    # 加载图片并转换为Tensor
    to_tensor = transforms.ToTensor()
    input_image = to_tensor(input_image).unsqueeze(0)  # [1, 3, H, W]
    ref_image = to_tensor(ref_image).unsqueeze(0)  # [1, 3, H, W]

    # 计算ref_image的均值和标准差（每个通道）
    mean_in = input_image.mean(dim=[0, 2, 3], keepdim=True)
    std_in = input_image.std(dim=[0, 2, 3], keepdim=True)
    mean = ref_image.mean(dim=[0, 2, 3], keepdim=True)
    std = ref_image.std(dim=[0, 2, 3], keepdim=True)

    # 用ref_image的均值和标准差对input_image进行标准化
    input_image = (input_image - mean_in) / std_in * std + mean

    to_pil = transforms.ToPILImage()
    input_image = to_pil(input_image.squeeze(0))  # 如果 tensor 形状是 [1, 3, H, W]，需要去掉 batch 维
    ref_image = to_pil(ref_image.squeeze(0))
    
    return input_image, ref_image


def init_latents_adain(input_image, ref_image):
    # 计算ref_image的均值和标准差（每个通道）
    mean_in = input_image.mean(dim=[0, 2, 3], keepdim=True)
    std_in = input_image.std(dim=[0, 2, 3], keepdim=True)
    mean = ref_image.mean(dim=[0, 2, 3], keepdim=True)
    std = ref_image.std(dim=[0, 2, 3], keepdim=True)

    # 用ref_image的均值和标准差对input_image进行标准化
    input_image = (input_image - mean_in) / std_in * std + mean
    
    return input_image

def save_cuda_tensor_as_pil(tensor, path):
    # tensor: [C, H, W] or [1, C, H, W], on cuda, range [0,1] or [0,255]
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.size(0) == 4:  # RGBA
        tensor = tensor[:3, ...]  # 只保留RGB
    tensor = tensor.mul(255).byte()
    img = tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    img_pil = Image.fromarray(img)
    img_pil.save(path)
    return img_pil



