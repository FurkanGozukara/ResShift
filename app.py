import argparse
from asyncio import tasks
from tkinter.tix import Form
import gradio as gr
from pathlib import Path
import os
import random
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from PIL import Image
import numpy as np
from utils import util_image
from basicsr.utils.download_util import load_file_from_url
import time

import platform

def open_folder():
    open_folder_path = os.path.abspath("results")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

os.environ['LOCAL_RANK'] = '0'
_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
}
_LINK = {
    'vqgan': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_inpainting_face_s4.pth',
    'faceir': 'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_faceir_s4.pth',
}

import cv2
from PIL import Image
import numpy as np

def upscale_if_needed(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        
    if width < 295 and height < 295:
        target_size = 295
        scale_factor = target_size / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        # Upscale using Lanczos
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Save the upscaled image
        upscaled_path = image_path.replace('.', '_upscaled.')
        cv2.imwrite(upscaled_path, upscaled)
        
        return upscaled_path
    
    return image_path

def get_configs(task='realsr', version='v3', scale=4):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if task == 'realsr':
        if version in ['v1', 'v2']:
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
        elif version == 'v3':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256_journal.yaml')
        else:
            raise ValueError(f"Unexpected version type: {version}")
        ckpt_url = _LINK[version]
        ckpt_path = ckpt_dir / f'resshift_{task}x{scale}_s{_STEP[version]}_{version}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif task == 'bicsr':
        configs = OmegaConf.load('./configs/bicx4_swinunet_lpips.yaml')
        ckpt_url = _LINK[task]
        ckpt_path = ckpt_dir / f'resshift_{task}x{scale}_s{_STEP[task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif task == 'faceir':
        configs = OmegaConf.load('./configs/faceir_gfpgan512_lpips.yaml')
        scale = 1
        ckpt_url = _LINK[task]
        ckpt_path = ckpt_dir / f'resshift_{task}_s{_STEP[task]}.pth'
        vqgan_url = _LINK['vqgan_face512']
        vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
    else:
        raise TypeError(f"Unexpected task type: {task}!")

    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    return configs

def predict(in_path, task='realsr', seed=12345, version='v3', randomize_seed=False):
    if randomize_seed:
        seed = random.randint(0, 1000000)

    in_path = upscale_if_needed(in_path)
    print("printing task")
    print(task)
    if task == 'faceir':
        scale = 1
        chop_size = 512
        chop_stride = 480
    else:
        scale = 4
        chop_size = 256
        chop_stride = 224

    configs = get_configs(task, version, scale)
    resshift_sampler = ResShiftSampler(
            configs,
            sf=scale,
            chop_size=chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    out_dir = Path('results')
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    im_sr, out_path = resshift_sampler.inference(
            in_path,
            out_dir,
            mask_path=None,
            bs=1,
            noise_repeat=False
            )

    # Convert numpy array to PIL Image for Gradio
    im_sr_pil = Image.fromarray((im_sr * 255).astype(np.uint8))

    return out_path, out_path, seed

def batch_process(input_folder, output_folder, task, seed, version, randomize_seed):
    if not output_folder:
        output_folder = 'results'
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob('*'))
    total_files = len(image_files)
    
    print(f"Found {total_files} images in the input folder.")
    
    start_time = time.time()
    for idx, img_file in enumerate(image_files, 1):
        file_start_time = time.time()
        print(f"Processing image {idx}/{total_files}: {img_file.name}")
        
        out_path, _, _ = predict(str(img_file), task, seed, version, randomize_seed)
        
        # Move the output file to the specified output folder
        new_out_path = os.path.join(output_path, os.path.basename(out_path))
        os.rename(out_path, new_out_path)
        
        file_end_time = time.time()
        file_duration = file_end_time - file_start_time
        
        print(f"Finished processing {img_file.name} in {file_duration:.2f} seconds")
        
        # Calculate and display estimated time remaining
        elapsed_time = file_end_time - start_time
        avg_time_per_file = elapsed_time / idx
        estimated_time_remaining = avg_time_per_file * (total_files - idx)
        
        print(f"Progress: {idx}/{total_files}")
        print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
        print("--------------------")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Batch processing completed. Total time: {total_duration:.2f} seconds")

title = "ResShift V2 by SECourses: Efficient Diffusion Model for Image Super-resolution by Residual Shifting"
description = r"""
Official repo : https://github.com/zsyOAOA/ResShift

# 1-Click Installers for Windows, RunPod & Massed Compute on : https://www.patreon.com/posts/110331752
"""
article = r"""

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResShift Gradio App")
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio app")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                in_image = gr.Image(label="Input: Low Quality Image", type="filepath")
                task = gr.Radio(
                    choices=[
                        ( "Real-world image super-resolution", "realsr"),
                        ( "Bicubic (resize by Matlab) image super-resolution","bicsr"),
                        ( "Blind Face Restoration","faceir")
                    ],
                    value="faceir",
                    label="Task"
                )
                gr.Markdown("Note: For 'faceir' task, scale is automatically set to 1.")
                with gr.Row():
                    seed = gr.Number(value=12345, precision=0, label="Random seed", step=1)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    version = gr.Dropdown(
                        choices=["v1", "v2", "v3"],
                        value="v3",
                        label="Version"
                    )
                btn = gr.Button("Generate")

            with gr.Column():
                out_image = gr.Image(label="Output: High Quality Image", format="png", type="filepath")         
                download_button = gr.File(label="Download the result")
                btn_open_outputs = gr.Button("Open Results Folder")
                btn_open_outputs.click(fn=open_folder)

        with gr.Row():
            batch_input_folder = gr.Textbox(label="Batch Input Folder Path")
            batch_output_folder = gr.Textbox(label="Batch Output Folder Path (optional)")
        
        batch_btn = gr.Button("Batch Process")

        examples = [
            ['./testdata/RealSet65/dog2.png', "realsr", 12345, "v3", False],
            ['./testdata/RealSet65/bears.jpg', "realsr", 12345, "v3", False],
            ['./testdata/RealSet65/oldphoto6.png', "realsr", 12345, "v3", False],
            ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00000067.png', "bicsr", 12345, "v3", False],
            ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00016898.png', "bicsr", 12345, "v3", False],
            ['./testdata/faceir/cropped_faces/lq/0143.png', "faceir", 12345, "v3", False],
            ['./testdata/faceir/cropped_faces/lq/0885.png', "faceir", 12345, "v3", False],
        ]

        gr.Examples(
            examples=examples,
            inputs=[in_image, task, seed, version, randomize_seed],
            outputs=[out_image, download_button, seed],
            fn=predict,
            cache_examples=False
        )

        btn.click(
            predict,
            inputs=[in_image, task, seed, version, randomize_seed],
            outputs=[out_image, download_button, seed],
            api_name="super_resolution"
        )

        batch_btn.click(
            batch_process,
            inputs=[batch_input_folder, batch_output_folder, task, seed, version, randomize_seed],
            outputs=None
        )

    demo.launch(inbrowser=True, share=args.share)