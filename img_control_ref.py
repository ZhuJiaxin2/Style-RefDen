import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionPipeline
from src.stable_diffusion_reference_only.pipelines.new_img_replace_kv_downandup_jumpattn1 import (
    StableDiffusionReplaceKVPipeline,
)
from src.stable_diffusion_reference_only.pipelines.new_img_reference import (
    StableDiffusionReferencePipeline,
)
from diffusers.utils import load_image
from utility.img2img_utils import img2img_with_reference, asian_canny_512
from utility.edge_detect_utils import edge_detector

input_image = Image.open("input/in.jpg").convert("RGB") 
ref_image = Image.open('style/starry-night.jpg').convert("RGB")

canny_image = edge_detector(input_image)
# canny_image = asian_canny_512(input_image)

controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", 
                                             torch_dtype=torch.float16)
pipe = StableDiffusionReferencePipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
).to('cuda:0')

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = img2img_with_reference(
                pipe,
                init_image=input_image,
                ref_image=ref_image,
                canny_image=canny_image,
                # prompt='1person',
                prompt='1person, clear facial features, best quality, masterpiece, (sharp, thick eyebrows:1.5), (well-defined nose bridge:1.3), clear nasal bridge with prominent bone highlights, enhance nose shadow, facial contour enhancement, nose shadow enhancement',
                negative_prompt='lowres, vague facial features, subtle nasal bridge, no eyebrows',
                strength=0.5,
                num_inference_steps=20,
                reference_attn=True,
                reference_adain=False,
                style_fid=0.8,
                ref_color_fid=1.0,
                width=1024,
                height=1024,
            ).images[0]  

result_img.save('output1.jpg')

print(f"gpu used {float(torch.cuda.max_memory_allocated(device=None)/1024**3)} GB")

def clean_cuda():
    import gc
    global pipe, controlnet
    for obj in [pipe, controlnet]:
        if obj is not None:
            del obj
    gc.collect()
    torch.cuda.empty_cache()
    
clean_cuda()
