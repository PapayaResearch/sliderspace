#!/bin/bash
cd "$(dirname "$0")/.."

python train_flux.py \
  --exp_name "art" \
  --num_sliders 64 \
  --clip_total_samples 50000 \
  --clip_batch_size 32 \
  --diverse_prompt_num 0 \
  --save_training_images true \
  --concept_prompts \
    "Renaissance oil painting, classical composition, rich colors, detailed figures" \
    "Baroque oil painting, dramatic chiaroscuro lighting, dynamic composition" \
    "Romantic painting, dramatic landscape, emotional intensity, stormy atmosphere" \
    "Impressionist painting, loose brushstrokes, natural outdoor light, dappled colors" \
    "Post-Impressionist painting, bold expressive colors, thick brushwork" \
    "Pointillist painting, small dots of color, vibrant scene" \
    "Cubist painting, fragmented geometric forms, multiple perspectives simultaneously" \
    "Surrealist painting, dreamlike bizarre imagery, uncanny juxtapositions" \
    "Abstract Expressionist painting, gestural brushstrokes, raw emotion, large canvas" \
    "Minimalist artwork, simple geometric forms, very limited color palette" \
    "Pop Art, bold flat colors, commercial imagery, graphic design influence" \
    "Art Nouveau illustration, organic flowing lines, decorative natural motifs" \
    "Expressionist painting, distorted forms, intense emotional colors" \
    "Color field painting, large areas of flat uniform color, subtle gradients" \
    "Contemporary abstract painting, mixed media, conceptual layered composition" \
    "Contemporary street art, urban graffiti style, bold outlines, spray paint" \
    "Contemporary digital art, modern aesthetic, vibrant colors, crisp edges" \
    "Contemporary conceptual artwork, thought-provoking, unconventional materials" \
    "Medieval illuminated manuscript style, gold leaf, flat decorative figures" \
    "Ancient fresco painting, muted earth tones, classical figures on wall" \
    "Japanese woodblock print, flat areas of color, decorative patterns, nature scene" \
    "Watercolor painting, translucent washes, soft edges, delicate" \
    "Charcoal drawing, dramatic shadows, expressive gestural lines" \
    "Engraving or etching print, fine detailed lines, black and white" \
    "Dark melancholic painting, somber muted tones, moody oppressive atmosphere" \
    "Joyful colorful artwork, bright vibrant palette, uplifting cheerful scene" \
    "Serene peaceful landscape painting, calm soft atmosphere, gentle light" \
    "Dramatic epic historical painting, monumental composition, heroic scene" \
    "Unsettling eerie artwork, disturbing imagery, dark surreal atmosphere" \
    "Classical still life oil painting, detailed objects, dramatic lighting" \
    "fine art black and white photography, dramatic contrast, artistic composition" \
    "long exposure fine art photography, light trails, ethereal motion blur" \
    "classical marble sculpture, carved stone, ancient Greek or Roman style" \
    "folk art naive painting, flat simple figures, bright colors, outsider art" \
    "intricate geometric patterns artwork, tessellations, symmetry, ornate decoration" \
    "Gothic art, dark medieval religious imagery, pointed arches, somber figures" \
    "Bauhaus Constructivist design, functional geometric shapes, primary colors, bold layout" \
    "photorealistic painting indistinguishable from a photograph, hyper-detailed" \
    "Chinese ink wash painting, sparse brushstrokes, negative space, misty mountains" \
    "African tribal art, bold patterns, masks, ceremonial motifs, earthy colors"
