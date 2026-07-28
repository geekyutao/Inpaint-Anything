# SAM 3 tracking + ProPainter inpainting. No tracker checkpoint needed.
python remove_anything_video.py \
    --input_video ./example/video/paragliding/original_video.mp4 \
    --coords_type key_in \
    --point_coords 652 162 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt

# Text-driven alternative: name the object in the first frame.
# python remove_anything_video.py \
#     --input_video ./example/video/paragliding/original_video.mp4 \
#     --text_select "paraglider" \
#     --dilate_kernel_size 15 \
#     --output_dir ./results

# Add --vi_fp16 if ProPainter runs out of GPU memory on long or large videos.

# Legacy stack (OSTrack tracking + STTN inpainting):
# python remove_anything_video.py \
#     --input_video ./example/video/paragliding/original_video.mp4 \
#     --coords_type key_in --point_coords 652 162 --point_labels 1 \
#     --dilate_kernel_size 15 --output_dir ./results \
#     --tracker ostrack --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
#     --vi_model sttn --vi_ckpt ./pretrained_models/sttn.pth --mask_idx 2
