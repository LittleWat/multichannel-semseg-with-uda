#!/usr/bin/env bash
#python tools/concat_rgb_gt_pred_img.py nyu  --pred_vis_dirs \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb_only_3ch---nyu-test_rgb/normal-drn_d_38-20.tar/vis \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha_only_3ch---nyu-test_hha/normal-drn_d_38-20.tar/vis \
python tools/concat_rgb_gt_pred_img.py nyu  --pick_up --pred_vis_dirs \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha_only_6ch---nyu-test_rgbhha/normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb2nyu-all_rgb_3ch---nyu-test_rgb/MCD-normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha2nyu-all_hha_3ch---nyu-test_hha/MCD-normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch_MFNet---nyu-test_rgbhha/MCD-MFNet-AddFusion-normal-drn_d_38-15/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/nyu-trainval_rgbhha_only_6ch---nyu-test_rgbhha/b8_512x512-drn_d_38-40.tar/vis