"""
Generate sample images from a list of prompts, the same way train_flux.py does
before the CLIP PCA step. Use this to inspect training data before committing
to a full training run.

Usage:
    python generate_samples.py \
        --concept_prompts "A realistic portrait of a person" "A realistic portrait of an elderly man" \
        --num_samples 50 \
        --output_dir sample_inspection/faces
"""

import argparse
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.inference_util import FluxPipelineSliders

from transformers import logging as hf_logging
from diffusers import logging as diff_logging
from utils.vlm_filter import approve_image, DEFAULT_VLM_PROMPT
hf_logging.set_verbosity_error()
diff_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_prompts", type=str, nargs="+", required=True,
                        help="List of prompts to sample from")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Total number of images to generate")
    parser.add_argument("--output_dir", type=str, default="sample_inspection",
                        help="Directory to save generated images")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images per prompt call")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    # VLM filter options
    parser.add_argument("--vlm_filter", action="store_true",
                        help="Enable VLM-based image quality filtering")
    parser.add_argument("--vlm_model", type=str, default="gemini/gemini-3-flash-preview",
                        help="LiteLLM model string to use for filtering")
    parser.add_argument("--vlm_prompt", type=str, default=None,
                        help="Custom approval prompt for the VLM (uses default if not set)")
    parser.add_argument("--vlm_workers", type=int, default=4,
                        help="Number of parallel workers for VLM filtering")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    weight_dtype = torch.bfloat16
    max_sequence_length = 256 if "schnell" in args.model_id.lower() else 512

    print(f"Loading model: {args.model_id}")
    pipe = FluxPipelineSliders.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    vlm_prompt = args.vlm_prompt or DEFAULT_VLM_PROMPT
    img_idx = 0
    rejected = 0
    pbar = tqdm(total=args.num_samples, desc="Generating")

    while img_idx < args.num_samples:
        prompt = random.choice(args.concept_prompts)
        images = pipe(
            prompt,
            num_images_per_prompt=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=max_sequence_length,
        ).images

        if args.vlm_filter:
            n_before = len(images)
            with ThreadPoolExecutor(max_workers=args.vlm_workers) as ex:
                results = list(ex.map(lambda img: approve_image(img, args.vlm_model, vlm_prompt), images))
            images = [img for img, ok in zip(images, results) if ok]
            n_rejected = n_before - len(images)
            if n_rejected:
                rejected += n_rejected
                pbar.set_postfix(rejected=rejected)

        for img in images:
            if img_idx >= args.num_samples:
                break
            img.save(os.path.join(args.output_dir, f"{img_idx:05d}.png"))
            img_idx += 1
            pbar.update(1)

    pbar.close()
    print(f"Saved {img_idx} images to: {args.output_dir}")


if __name__ == "__main__":
    main()
