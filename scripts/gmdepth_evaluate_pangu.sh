#!/usr/bin/env bash


# gmdepth-scale1-regrefine1
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--output_path /home/leonwilliams/workshop/pangu/panguData/flight00_results_unimatch_depth \
--inference_dir /home/leonwilliams/workshop/pangu/panguData/flight00test \
--resume pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth \
--reg_refine \
--num_reg_refine 1

# # gmdepth-scale1-regrefine1
# CUDA_VISIBLE_DEVICES=0 python main_depth.py \
# --eval \
# --output_path /home/leonwilliams/workshop/pangu/panguData/flight00_results_unimatch_depth \
# --inference_dir /home/leonwilliams/workshop/pangu/panguData/flight00test \
# --resume pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth \
# --val_dataset demon \
# --demon_split scenes11 \
# --reg_refine \
# --num_reg_refine 1

