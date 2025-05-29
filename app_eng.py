import sys
import os

# 将项目根目录添加到 PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import huggingface_hub
import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionPipeline
from src.stable_diffusion_reference_only.pipelines.new_img_replace_kv_downandup import (
    StableDiffusionReplaceKVPipeline,
)
from src.stable_diffusion_reference_only.pipelines.new_img_reference import (
    StableDiffusionReferencePipeline,
)
from utility.img2img_utils import get_canny_img, img2img_with_reference, europe_get_enhanced_edges
from utility.segment_utils import portrait_segment

from anime_segmentation import get_model as get_anime_segmentation_model
from anime_segmentation import character_segment as anime_character_segment

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# pipe2 = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1-base",
#         safety_checker=None,
#         torch_dtype=torch.float16
# ).to('cuda:0')

# # ref_image = pipe2(
# #     prompt='pop art of night city',
# #     height=512,
# #     width=512,
# #     num_inference_steps=20,
# # ).images[0]
# # ref_image.save('ref.jpg')

# torch.cuda.empty_cache()

segment_model = get_anime_segmentation_model(
    model_path=huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.ckpt")
).to(device)

def character_segment(img):
    if img is None:
        return None
    img = anime_character_segment(segment_model, img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def color_inversion(img):
    if img is None:
        return None
    return 255 - img


def get_line_art(img):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=5,
        C=7,
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def inference(prompt, input_image, strength, num_inference_steps, style_fid, model_select):
    if prompt is None or input_image is None:
        return None
    
    original_width, original_height = input_image.size
    ref_image = prompt
    canny_image = europe_get_enhanced_edges(input_image)
    # canny_image = get_canny_img(canny_image)

    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", 
                                         torch_dtype=torch.float16)
    
    if model_select == "attn_only (most content preserved)":
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
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        reference_attn=True,
                        reference_adain=False,
                        style_fid=style_fid,
                    ).images[0]  

        return np.array(
                result_img.resize((original_width, original_height)),
            )
        
    elif model_select == "attn+adain (most style artistic)":
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
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        reference_attn=True,
                        reference_adain=True,
                        style_fid=style_fid,
                    ).images[0]  

        return np.array(
                result_img.resize((original_width, original_height)),
            )
    
    elif model_select == "replace_kv (most balanced)":
        pipe = StableDiffusionReplaceKVPipeline.from_pretrained(
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
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        reference_attn=True,
                        reference_adain=True,
                        style_fid=style_fid,
                    ).images[0]  

        return np.array(
                result_img.resize((original_width, original_height)),
            )


def automatic_coloring(prompt, blueprint, num_inference_steps):
    if prompt is None or blueprint is None:
        return None
    blueprint = color_inversion(blueprint)
    return inference(prompt, blueprint, num_inference_steps)


def style_transfer(prompt, blueprint, num_inference_steps):
    if prompt is None or blueprint is None:
        return None
    prompt = character_segment(prompt)
    blueprint = character_segment(blueprint)
    blueprint = get_line_art(blueprint)
    blueprint = color_inversion(blueprint)
    return inference(prompt, blueprint, num_inference_steps)

def resize(img, new_height, new_width):
    img = img.resize((int(new_width), int(new_height)), Image.LANCZOS)
    return np.array(img)

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 华中科技大学本科生毕业设计\n\n
    demo by 祝家心
    """
    )
    with gr.Row():
        with gr.Column():
            style_prompt_compoent = gr.Image(label="style image", type="pil")
            with gr.Row():
                prompt_new_height = gr.Number(512, label="height", minimum=1)
                prompt_new_width = gr.Number(512, label="width", minimum=1)
                prompt_resize_button = gr.Button("style image resize")
                prompt_resize_button.click(
                    resize,
                    inputs=[
                        style_prompt_compoent,
                        prompt_new_height,
                        prompt_new_width,
                    ],
                    outputs=style_prompt_compoent,
                )
            # text_prompt_component = gr.Textbox(label="style text prompt")
            
        with gr.Column():
            input_image_compoent = gr.Image(label="input image", type="pil")

            with gr.Row():
                input_new_height = gr.Number(512, label="height", minimum=1)
                input_new_width = gr.Number(512, label="width", minimum=1)
                with gr.Column():
                    input_resize_button = gr.Button("input resize")
                    input_resize_button.click(
                        resize,
                        inputs=[
                            input_image_compoent,
                            input_new_height,
                            input_new_width,
                        ],
                        outputs=input_image_compoent,
                    )
                    input_portrait_crop_button = gr.Button("portrait crop")
                    input_portrait_crop_button.click(
                        portrait_segment,
                        inputs=[
                            input_image_compoent,
                        ],
                        outputs=input_image_compoent,
                    )
    
        with gr.Column():
            result_output_component = gr.Image(label="result", type="pil")
            gr.Markdown('''If strength=0.5 for 20 steps, inference takes about 30 seconds.\n
                        If your face is distorted, lower style fidelity (the robust value is 0.1).\n
                        If you want a more artistic style, increase style fidelity (the balanced value is 0.2-0.3, the artistic value is 0.5).''')
            strength_input_component = gr.Number(
                0.5, label="strength", minimum=0, maximum=1
            )
            num_inference_steps_input_component = gr.Number(
                20, label="num inference steps", minimum=1, maximum=1000, step=1
            )
            style_fid_input_component = gr.Number(
                0.1, label="style fidelity", minimum=0, maximum=1, 
                info="style fidelity of ref_uncond_xt. If style_fidelity=1.0, reference style more important, elif style_fidelity=0.0, prompt more important, else balanced."
            )
            
            model_options = [
                "attn+adain (most style artistic)",
                "attn_only (most content preserved)",
                "replace_kv (most balanced)"
            ]
            # CheckboxGroup示例（多选）
            # gr.Markdown("### CheckboxGroup示例（多选）")
            model_select = gr.Radio(
                choices=model_options,
                label="model selection",
                value="attn+adain (most style artistic)", 
                # info='''attn+adain: most style artistic; 
                #         attn_only: most content preserved; 
                #         replace_kv: most balanced for content and style.'''
            )
            
            inference_button = gr.Button("inference")
            inference_button.click(
                inference,
                inputs=[
                    style_prompt_compoent,
                    input_image_compoent,
                    strength_input_component,
                    num_inference_steps_input_component,
                    style_fid_input_component,
                    model_select,
                ],
                outputs=result_output_component,
            )

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    os.path.join(
                        os.path.dirname(__file__), "style", "starry-night.jpg"
                    ),
                    os.path.join(
                        os.path.dirname(__file__), "input", "in.jpg"
                    ),
                ],
            ],
            inputs=[style_prompt_compoent, input_image_compoent],
            # outputs=[],
            # outputs=result_output_component,
            # fn=lambda x, y: None,
            # cache_examples=True,
        )
if __name__ == "__main__":
    demo.launch(server_port=6006)
