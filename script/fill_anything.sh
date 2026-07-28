python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt

# Text-driven alternative (SAM 3 only): --text_select picks the target,
# --text_prompt still describes what to generate in its place.
# python fill_anything.py \
#     --input_img ./example/fill-anything/sample1.png \
#     --text_select "dog" \
#     --text_prompt "a teddy bear on a bench" \
#     --dilate_kernel_size 50 \
#     --output_dir ./results
