

export CUDA_VISIBLE_DEVICES="3"

python segment_inpaint_one_img.py \
--input_img ./example/remove-anything/dog.jpg \
--point_coords 200 450 \
--point_labels 1 \
--output_dir ./example \
--sam_model_type "vit_h" \
--sam_ckpt /data0/fengrs/compression2023/Inpaint-Anything/sam/sam_vit_h_4b8939.pth \
--lama_ckpt /data1/yutao/projects/IAM/lama/big-lama