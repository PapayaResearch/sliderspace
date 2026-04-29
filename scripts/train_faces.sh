#!/bin/bash
cd "$(dirname "$0")/.."

python train_flux.py \
  --exp_name "faces" \
  --num_sliders 64 \
  --clip_total_samples 50000 \
  --clip_batch_size 32 \
  --diverse_prompt_num 0 \
  --save_training_images true \
  --concept_prompts \
    "id photo of a person from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a girl from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a boy from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage White European man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage White European woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult White European man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult White European woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged White European man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged White European woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly White European man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly White European woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Black African man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Black African woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Black African man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Black African woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Black African man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Black African woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Black African man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Black African woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage East Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage East Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult East Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult East Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged East Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged East Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly East Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly East Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage South Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage South Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult South Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult South Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged South Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged South Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly South Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly South Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Southeast Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Southeast Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Southeast Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Southeast Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Southeast Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Southeast Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Southeast Asian man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Southeast Asian woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Middle Eastern man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Middle Eastern woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Middle Eastern man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Middle Eastern woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Middle Eastern man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Middle Eastern woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Middle Eastern man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Middle Eastern woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Hispanic Latino man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Hispanic Latino woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Hispanic Latino man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Hispanic Latino woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Hispanic Latino man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Hispanic Latino woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Hispanic Latino man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Hispanic Latino woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Indigenous Native American man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a teenage Indigenous Native American woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Indigenous Native American man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a young adult Indigenous Native American woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Indigenous Native American man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a middle-aged Indigenous Native American woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Indigenous Native American man from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of an elderly Indigenous Native American woman from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a happy expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a sad expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with an angry expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a surprised expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a neutral expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a disgusted expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a fearful expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a contemptuous expression from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person laughing from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person crying from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with very light skin from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with light skin from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with medium skin from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with dark skin from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with very dark skin from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with short straight hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with long straight hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with short curly hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with long curly hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with coily hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with wavy hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with black hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with brown hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with blonde hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with red hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with grey hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with white hair from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a beard from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a mustache from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with stubble from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a bald person from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with an oval face from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a round face from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a square face from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a narrow face from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing glasses from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing sunglasses from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing a hijab from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing a turban from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing a hat from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with light makeup from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with heavy makeup from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with earrings from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with facial piercings from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a subtle neck tattoo from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with prominent facial tattoos from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with freckles from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with vitiligo from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with acne from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a slim person from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a heavyset person from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person wearing hearing aids from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with a facial difference from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face" \
    "id photo of a person with Down syndrome from the shoulders up, plain white background, looking straight at camera, even lighting, centered, eye level camera angle, unobstructed face"
