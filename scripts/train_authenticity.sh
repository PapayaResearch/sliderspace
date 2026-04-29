#!/bin/bash
cd "$(dirname "$0")/.."

SCENE="a narrow alley flanked by Victorian red brick buildings with arched windows, wet cobblestone ground, a green wooden door on the left wall, an iron streetlamp above, a wrought iron gate at the far end with a foggy street beyond"

python train_flux.py \
  --exp_name "authenticity_v2" \
  --num_sliders 64 \
  --clip_total_samples 50000 \
  --clip_batch_size 32 \
  --diverse_prompt_num 0 \
  --save_training_images true \
  --concept_prompts \
    "CCTV security camera footage of ${SCENE}, grainy high-angle wide-angle shot, timestamp overlay, washed out colors" \
    "smartphone photo of ${SCENE} taken at arm's length, slight wide-angle distortion, casual framing" \
    "smartphone snapshot of ${SCENE}, casual handheld shot, slightly off-center framing, mobile photography" \
    "disposable film camera photo of ${SCENE}, overexposed flash, light leaks, grainy, washed out colors" \
    "sharp professional photograph of ${SCENE}, shallow depth of field, blurred background bokeh, high resolution" \
    "dashcam footage of ${SCENE}, road perspective through windshield, wide angle, slightly fisheye" \
    "bodycam footage of ${SCENE}, chest-level perspective, slight motion blur" \
    "raw unedited photograph of ${SCENE}, natural colors, no post-processing, flat exposure" \
    "heavily retouched photograph of ${SCENE}, enhanced colors, airbrushed appearance, magazine style" \
    "cinematic color graded photograph of ${SCENE}, teal and orange tones, film look" \
    "vintage film filter photograph of ${SCENE}, faded colors, film grain, retro look" \
    "HDR photograph of ${SCENE}, high dynamic range, enhanced detail in highlights and shadows" \
    "photograph of ${SCENE} saved at very low JPEG quality, blocky compression artifacts visible, degraded" \
    "analog film grain photograph of ${SCENE}, textured grain, warm tones, film photography" \
    "AI-generated image of ${SCENE}, uncanny realism, slight surreal quality, synthetic appearance" \
    "natural light photograph of ${SCENE}, soft outdoor lighting, no flash, available light" \
    "harsh artificial light photograph of ${SCENE}, strong shadows, fluorescent or neon lighting" \
    "commercial advertising photograph of ${SCENE}, polished and stylized, studio quality" \
    "photojournalism documentary photograph of ${SCENE}, candid moment, high contrast, reportage style" \
    "composited photograph of ${SCENE} with a fake background, cutout edges visible, inconsistent lighting" \
    "Polaroid instant photo of ${SCENE}, square format, faded colors, white border, slightly overexposed" \
    "long exposure photograph of ${SCENE}, motion blur on moving elements, sharp background, slow shutter speed" \
    "black and white street photograph of ${SCENE}, high contrast, candid, silver gelatin film look"
