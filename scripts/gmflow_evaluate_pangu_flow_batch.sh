#!/usr/bin/env bash

# gmflow-scale2-regrefine6, inference on image dir
CUDA_VISIBLE_DEVICES=3 nohup python main_flow.py \
--inference_dir /home/leonwilliams/workshop/pangu/panguData/originalFlight/flight08 \
--resume pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
--output_path /home/leonwilliams/workshop/pangu/panguData/originalFlight/flight08/flight08_results_unimatch_v2 \
--padding_factor 32 \
--upsample_factor 4 \
--save_flo_flow \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 &

CUDA_VISIBLE_DEVICES=5 nohup python main_flow.py \
--inference_dir /home/leonwilliams/workshop/pangu/panguData/originalFlight/flight00 \
--resume pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
--output_path /home/leonwilliams/workshop/pangu/panguData/originalFlight/flight00/flight00_results_unimatch_v2 \
--padding_factor 32 \
--upsample_factor 4 \
--save_flo_flow \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 &

CUDA_VISIBLE_DEVICES=7 nohup python main_flow.py \
--inference_dir /home/leonwilliams/workshop/pangu/panguData/horizontalTestFlight/test00 \
--resume pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
--output_path /home/leonwilliams/workshop/pangu/panguData/horizontalTestFlight/test00/test00_results_unimatch \
--padding_factor 32 \
--upsample_factor 4 \
--save_flo_flow \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 &

CUDA_VISIBLE_DEVICES=8 nohup python main_flow.py \
--inference_dir /home/leonwilliams/workshop/pangu/panguData/horizontalTestFlight/test00_reshaped \
--resume pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
--output_path /home/leonwilliams/workshop/pangu/panguData/horizontalTestFlight/test00_reshaped/test00_reshaped_results_unimatch \
--padding_factor 32 \
--upsample_factor 4 \
--save_flo_flow \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 &
