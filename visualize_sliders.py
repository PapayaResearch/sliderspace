"""
Visualize trained SliderSpace directions for FLUX or SDXL.
Saves a grid image per slider: rows are interpolation steps (scale 0 → slider_scale),
columns are different seeds.

Usage:
    # FLUX (default)
    python visualize_sliders.py \
        --sliderspace_path trained_sliders/flux/faces \
        --prompt "picture of a human face" \
        --output_dir slider_visualizations/faces

    # SDXL
    python visualize_sliders.py \
        --model_type sdxl \
        --sliderspace_path trained_sliders/sdxl/face \
        --prompt "picture of a human face" \
        --output_dir slider_visualizations/faces_sdxl

    # With VLM filtering:
    python visualize_sliders.py \
        --sliderspace_path trained_sliders/flux/faces \
        --prompt "picture of a human face" \
        --output_dir slider_visualizations/faces \
        --vlm_filter
"""

import argparse
import glob
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.lora import DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV, LoRANetwork
from utils.inference_util import FluxPipelineSliders, StableDiffusionXLPipelineSliders
from utils.vlm_filter import approve_image, DEFAULT_VLM_PROMPT

from transformers import logging as hf_logging
from diffusers import logging as diff_logging
hf_logging.set_verbosity_error()
diff_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="flux", choices=["flux", "sdxl"],
                        help="Model type to use for visualization")
    parser.add_argument("--sliderspace_path", type=str, required=True,
                        help="Directory containing trained slider .pt files")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to use for visualization")
    parser.add_argument("--output_dir", type=str, default="slider_visualizations",
                        help="Where to save output images")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Model ID (defaults to FLUX.1-schnell or SDXL-base depending on model_type)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_images", type=int, default=6,
                        help="Number of columns in the grid (different seeds)")
    parser.add_argument("--num_interpolation_steps", type=int, default=5,
                        help="Number of rows in the grid (scale 0 to slider_scale)")
    parser.add_argument("--slider_scale", type=float, default=1.0,
                        help="Maximum scale to apply the slider at")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of inference steps (defaults to 4 for FLUX, 4 for SDXL-DMD2)")
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--apply_sliders_from", type=int, default=0,
                        help="Increase for more precise edits (try 0 or 1)")
    parser.add_argument("--train_method", type=str, default=None,
                        help="LoRA train method (defaults to flux-attn or xattn-strict depending on model_type)")
    parser.add_argument("--slider_rank", type=int, default=1)
    parser.add_argument("--slider_alpha", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for reproducibility (random if not set)")
    # VLM filter options
    parser.add_argument("--vlm_filter", action="store_true",
                        help="Enable VLM-based filtering of baseline images")
    parser.add_argument("--vlm_model", type=str, default="gemini/gemini-3-flash-preview",
                        help="LiteLLM model string to use for filtering")
    parser.add_argument("--vlm_prompt", type=str, default=None,
                        help="Custom approval prompt for the VLM (uses default if not set)")
    parser.add_argument("--vlm_workers", type=int, default=4,
                        help="Number of parallel workers for VLM filtering")
    return parser.parse_args()


def load_pipeline(args, weight_dtype):
    if args.model_type == "flux":
        model_id = args.model_id or "black-forest-labs/FLUX.1-schnell"
        print(f"Loading FLUX model: {model_id}")
        pipe = FluxPipelineSliders.from_pretrained(model_id, torch_dtype=weight_dtype)
        pipe = pipe.to(args.device)
        backbone = pipe.transformer
    else:
        from diffusers import UNet2DConditionModel, LCMScheduler
        from huggingface_hub import hf_hub_download
        model_id = args.model_id or "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Loading SDXL model: {model_id}")
        unet = UNet2DConditionModel.from_config(model_id, subfolder="unet").to(args.device, weight_dtype)
        unet.load_state_dict(torch.load(hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_4step_unet_fp16.bin")))
        pipe = StableDiffusionXLPipelineSliders.from_pretrained(model_id, unet=unet, torch_dtype=weight_dtype)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(args.device).to(weight_dtype)
        backbone = pipe.unet
    return pipe, backbone


def run_pipe(pipe, args, seed, networks, max_sequence_length):
    generator = torch.manual_seed(seed)
    kwargs = dict(
        num_images_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        networks=networks,
        apply_sliders_from=args.apply_sliders_from,
        apply_sliders_till=None,
    )
    if args.model_type == "flux":
        kwargs["max_sequence_length"] = max_sequence_length
    return pipe(args.prompt, **kwargs).images[0]


def make_grid(rows):
    """Grid where each row is a list of images. Rows = interpolation steps, cols = seeds."""
    w, h = rows[0][0].size
    n_rows = len(rows)
    n_cols = len(rows[0])
    grid = Image.new("RGB", (w * n_cols, h * n_rows))
    for row_idx, row in enumerate(rows):
        for col_idx, img in enumerate(row):
            grid.paste(img, (col_idx * w, row_idx * h))
    return grid


def get_good_seeds(pipe, args, networks, max_sequence_length, vlm_prompt):
    """Generate baseline images, optionally filter with VLM, return (seeds, images)."""
    good_seeds = []
    good_images = []
    next_seed = args.seed if args.seed is not None else random.randint(0, 2**15)
    rejected = 0

    networks[0].set_lora_slider(0.0)
    pipe.set_progress_bar_config(disable=True)

    while len(good_seeds) < args.num_images:
        batch_seeds = list(range(next_seed, next_seed + args.num_images))
        next_seed += args.num_images
        batch_images = [run_pipe(pipe, args, s, networks, max_sequence_length) for s in batch_seeds]

        if args.vlm_filter:
            with ThreadPoolExecutor(max_workers=args.vlm_workers) as ex:
                approved = list(ex.map(lambda img: approve_image(img, args.vlm_model, vlm_prompt), batch_images))
            batch_seeds = [s for s, ok in zip(batch_seeds, approved) if ok]
            batch_images = [img for img, ok in zip(batch_images, approved) if ok]
            rejected += args.num_images - len(batch_seeds)
            if rejected:
                print(f"  VLM rejected {rejected} baseline images so far")

        for s, img in zip(batch_seeds, batch_images):
            if len(good_seeds) >= args.num_images:
                break
            good_seeds.append(s)
            good_images.append(img)

    pipe.set_progress_bar_config(disable=False)
    return good_seeds, good_images


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_method is None:
        args.train_method = "flux-attn" if args.model_type == "flux" else "xattn-strict"
    if args.num_inference_steps is None:
        args.num_inference_steps = 4

    weight_dtype = torch.bfloat16
    model_id = args.model_id or ("black-forest-labs/FLUX.1-schnell" if args.model_type == "flux" else "stabilityai/stable-diffusion-xl-base-1.0")
    max_sequence_length = 256 if "schnell" in model_id.lower() else 512

    pipe, backbone = load_pipeline(args, weight_dtype)

    network = LoRANetwork(
        backbone,
        rank=args.slider_rank,
        multiplier=1.0,
        alpha=args.slider_alpha,
        train_method=args.train_method,
        fast_init=True,
    ).to(args.device, dtype=weight_dtype)
    networks = {0: network}

    sliders = sorted(glob.glob(os.path.join(args.sliderspace_path, "*.pt")))
    if not sliders:
        print(f"No .pt files found in {args.sliderspace_path}")
        sys.exit(1)
    print(f"Found {len(sliders)} sliders")

    vlm_prompt = args.vlm_prompt or DEFAULT_VLM_PROMPT
    scales = np.linspace(0, args.slider_scale, args.num_interpolation_steps)

    print("Generating baseline images...")
    seeds, baseline_images = get_good_seeds(pipe, args, networks, max_sequence_length, vlm_prompt)

    for slider_path in tqdm(sliders, desc="Sliders"):
        slider_name = os.path.splitext(os.path.basename(slider_path))[0]
        networks[0].load_state_dict(torch.load(slider_path, weights_only=False))

        rows = [baseline_images]
        pipe.set_progress_bar_config(disable=True)
        for scale in scales[1:]:
            networks[0].set_lora_slider(float(scale))
            row = [run_pipe(pipe, args, s, networks, max_sequence_length) for s in seeds]
            rows.append(row)
        pipe.set_progress_bar_config(disable=False)

        grid = make_grid(rows)
        grid.save(os.path.join(args.output_dir, f"{slider_name}.png"))

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
