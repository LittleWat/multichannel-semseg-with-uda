#!/usr/bin/env bash
#python tools/concat_rgb_gt_pred_img.py nyu  --pred_vis_dirs \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb_only_3ch---nyu-test_rgb/normal-drn_d_38-20.tar/vis \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha_only_3ch---nyu-test_hha/normal-drn_d_38-20.tar/vis \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb2nyu-all_rgb_3ch---nyu-test_rgb/MCD-normal-drn_d_38-20.tar/vis \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha2nyu-all_hha_3ch---nyu-test_hha/MCD-normal-drn_d_38-20.tar/vis \
python tools/concat_multitask_vis_results.py nyu  --pick_up --n_img 8 --pred_vis_dirs \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha_only_6ch---nyu-test_rgbhha/normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch_MFNet---nyu-test_rgbhha/MCD-MFNet-AddFusion-normal-drn_d_38-15/vis \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/depth \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/boundary \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/refined_vis \
