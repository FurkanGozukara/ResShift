import argparse
from asyncio import tasks
from tkinter.tix import Form
import gradio as gr
from pathlib import Path
import os
import random
from pathlib import Path
import os
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from PIL import Image
import numpy as np
from utils import util_image
from basicsr.utils.download_util import load_file_from_url

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

    print("printing  task ")
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

title = "ResShift V1 by SECourses: Efficient Diffusion Model for Image Super-resolution by Residual Shifting"
description = r"""
Official repo : https://github.com/zsyOAOA/ResShift

# 1-Click Installers for Windows, RunPod & Massed Compute on : https://www.patreon.com/posts/110331752
"""
article = r"""

"""

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
            download_button = gr.File(label="Download the output")

    examples = [
        ['./testdata/RealSet65/dog2.png', "realsr", 12345, "v3", False],
        ['./testdata/RealSet65/bears.jpg', "realsr", 12345, "v3", False],
        ['./testdata/RealSet65/oldphoto6.png', "realsr", 12345, "v3", False],
        ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00000067.png', "bicsr", 12345, "v3", False],
        ['./testdata/Bicubicx4/lq_matlab/ILSVRC2012_val_00016898.png', "bicsr", 12345, "v3", False],
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

if __name__ == "__main__":
    demo.launch(inbrowser=True)