python replace_anything.py \
    --input_img ./example/replace-anything/dog.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results \
    --sam_model_type "vit_t" \
    --sam_ckpt ./weights/mobile_sam.pt