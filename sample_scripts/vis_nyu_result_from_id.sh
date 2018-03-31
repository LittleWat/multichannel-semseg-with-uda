#!/usr/bin/env bash

python tools/visualize_result.py  --outdir vis_nyu \
--title_names RGB HHA GT Oracle SrcOnly_RGBHHA Adapt_RGB Adapt_HHA Adapt_EarlyFusion Adapt_LateFusion Adapt_Multitask  \
--indir_list \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/nyu-trainval_rgbhha_only_6ch---nyu-test_rgbhha/b8_512x512-drn_d_38-40.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha_only_6ch---nyu-test_rgbhha/normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgb2nyu-all_rgb_3ch---nyu-test_rgb/MCD-normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_hha2nyu-all_hha_3ch---nyu-test_hha/MCD-normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch_MFNet---nyu-test_rgbhha/MCD-MFNet-AddFusion-normal-drn_d_38-15/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-trainval_rgbhha_6ch_MCDmultitask---nyu-test_rgbhha/MCD-normal-drn_d_38-40.tar/label \
