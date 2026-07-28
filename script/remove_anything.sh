python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --coords_type key_in \
    --point_coords 200 450 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama

# Text-driven alternative (SAM 3 only): no coordinates needed.
# python remove_anything.py \
#     --input_img ./example/remove-anything/dog.jpg \
#     --text_select "dog" \
#     --dilate_kernel_size 15 \
#     --output_dir ./results \
#     --lama_ckpt ./pretrained_models/big-lama
