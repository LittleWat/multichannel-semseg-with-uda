#!/usr/bin/env bash
#python tools/concat_rgb_gt_pred_img.py nyu  --pred_vis_dirs \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb_only_3ch---nyu-test_rgb/normal-drn_d_38-20.tar/vis \
#/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha_only_3ch---nyu-test_hha/normal-drn_d_38-20.tar/vis \
python tools/concat_multitask_vis_results.py nyu  --pick_up --pred_vis_dirs \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/depth \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/boundary \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/vis \
/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/refined_vis \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/nyu-trainval_rgbhha_only_6ch---nyu-test_rgbhha/b4_1024x1024-drn_d_38-40.tar/vis