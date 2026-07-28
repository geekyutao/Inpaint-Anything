python replace_anything.py \
    --input_img ./example/replace-anything/dog.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results \
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt

# Text-driven alternative (SAM 3 only): --text_select picks the foreground to
# keep, --text_prompt describes the new background.
# python replace_anything.py \
#     --input_img ./example/replace-anything/dog.png \
#     --text_select "dog" \
#     --text_prompt "sit on the swing" \
#     --output_dir ./results
