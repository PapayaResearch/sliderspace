#!/bin/bash
cd "$(dirname "$0")/.."

python train_flux.py \
  --exp_name "affordances" \
  --num_sliders 64 \
  --clip_total_samples 50000 \
  --clip_batch_size 32 \
  --diverse_prompt_num 0 \
  --save_training_images true \
  --concept_prompts \
    "a photograph of a small shiny metal object, plain white background" \
    "a photograph of a rough wooden tool, plain white background" \
    "a photograph of a transparent glass container, plain white background" \
    "a photograph of a colorful plastic object, plain white background" \
    "a photograph of a soft fabric pouch or bag, plain white background" \
    "a photograph of a rubber or silicone flexible object, plain white background" \
    "a photograph of a ceramic or porcelain container, plain white background" \
    "a photograph of a heavy stone or rock object, plain white background" \
    "a photograph of a paper or cardboard object, plain white background" \
    "a photograph of a leather object, plain white background" \
    "a photograph of a round spherical object, plain white background" \
    "a photograph of a flat thin disc-shaped object, plain white background" \
    "a photograph of a long cylindrical object, plain white background" \
    "a photograph of a sharp pointed object, plain white background" \
    "a photograph of a cubic box-shaped object, plain white background" \
    "a photograph of an irregular asymmetric object, plain white background" \
    "a photograph of a ring or hoop-shaped object, plain white background" \
    "a photograph of a curved arched object, plain white background" \
    "a photograph of a tiny miniature object that fits on a fingertip, plain white background" \
    "a photograph of a small pocket-sized object, plain white background" \
    "a photograph of a large heavy bulky object, plain white background" \
    "a photograph of an object with a long handle, plain white background" \
    "a photograph of an object with a looped handle, plain white background" \
    "a photograph of an object with a grip or knob, plain white background" \
    "a photograph of a smooth object with no handle or grip, plain white background" \
    "a photograph of a hollow container with a wide opening, plain white background" \
    "a photograph of a hollow container with a narrow spout, plain white background" \
    "a photograph of a solid object with no openings, plain white background" \
    "a photograph of an object with multiple holes or perforations, plain white background" \
    "a photograph of a stackable flat object, plain white background" \
    "a photograph of a collapsible or foldable object, plain white background" \
    "a photograph of a very smooth polished object, plain white background" \
    "a photograph of a rough heavily textured object, plain white background" \
    "a photograph of a soft padded object, plain white background" \
    "a photograph of a sharp-edged angular object, plain white background" \
    "a photograph of a hand tool like a wrench or hammer, plain white background" \
    "a photograph of a kitchen utensil, plain white background" \
    "a photograph of a container with a lid, plain white background" \
    "a photograph of a writing instrument, plain white background" \
    "a photograph of a fastener like a bolt, screw or clip, plain white background" \
    "a photograph of a natural object like a shell, seed or stone, plain white background" \
    "a photograph of an electronic device or gadget, plain white background" \
    "a photograph of a woven or braided object, plain white background" \
    "a photograph of a sharp cutting tool, plain white background" \
    "a photograph of a rope or cord-like object, plain white background" \
    "a photograph of a flat plate or tray, plain white background" \
    "a photograph of a measuring or gripping tool, plain white background" \
    "a photograph of a jointed or hinged object, plain white background" \
    "a photograph of a transparent object you can see through, plain white background" \
    "a photograph of a hollow object with a narrow neck, plain white background" \
    "a photograph of a small metal object with a long thin handle, plain white background" \
    "a photograph of a large wooden flat object with no openings, plain white background" \
    "a photograph of a transparent glass object with a narrow neck and wide base, plain white background" \
    "a photograph of a soft fabric object with a looped handle, plain white background" \
    "a photograph of a small ceramic container with a wide opening and no lid, plain white background" \
    "a photograph of a heavy metal object with multiple holes, plain white background" \
    "a photograph of a flexible rubber object with a cylindrical shape, plain white background" \
    "a photograph of a small smooth spherical metal object, plain white background" \
    "a photograph of a large hollow plastic container with a lid, plain white background" \
    "a photograph of a flat wooden object with a handle on one end, plain white background" \
    "a photograph of a sharp metal object with a pointed tip and a grip, plain white background" \
    "a photograph of a small plastic object with multiple buttons or controls, plain white background" \
    "a photograph of a woven object with an open basket shape, plain white background" \
    "a photograph of a long thin wooden cylindrical object, plain white background" \
    "a photograph of a heavy stone flat rectangular object, plain white background" \
    "a photograph of a metal coiled spiral object, plain white background" \
    "a photograph of a small leather flat foldable object, plain white background" \
    "a photograph of a ceramic object with a handle and a spout, plain white background" \
    "a photograph of a metal hinged object with two arms, plain white background" \
    "a photograph of a soft padded fabric object with a zipper, plain white background" \
    "a photograph of a transparent plastic object with a narrow tip, plain white background" \
    "a photograph of a rough stone object with an irregular pointed shape, plain white background" \
    "a photograph of a large cylindrical metal container with a lid, plain white background" \
    "a photograph of a small wooden object with an irregular carved shape, plain white background" \
    "a photograph of a flat perforated metal object with a long handle, plain white background" \
    "a photograph of a glass spherical object with a smooth polished surface, plain white background" \
    "a photograph of a rubber object with a bulbous hollow squeezable shape, plain white background" \
    "a photograph of a metal object with a threaded cylindrical shape, plain white background" \
    "a photograph of a large fabric object that collapses flat, plain white background" \
    "a photograph of a small electronic object with a flat rectangular shape and a screen, plain white background"
