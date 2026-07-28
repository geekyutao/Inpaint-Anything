python remove_anything_3d.py \
    --input_dir ./example/3d/horns \
    --coords_type key_in \
    --point_coords 830 405 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --mask_idx 1 \
    --config ./nerf/configs/horns.txt \
    --expname horns

# Text-driven alternative (SAM 3 only): the phrase picks the target in the
# first view, OSTrack then follows it across the remaining views.
# python remove_anything_3d.py \
#     --input_dir ./example/3d/horns \
#     --text_select "horns" \
#     --dilate_kernel_size 15 \
#     --output_dir ./results \
#     --lama_ckpt ./pretrained_models/big-lama \
#     --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
#     --config ./nerf/configs/horns.txt \
#     --expname horns
