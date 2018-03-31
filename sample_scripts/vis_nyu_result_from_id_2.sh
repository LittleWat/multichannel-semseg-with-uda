#!/usr/bin/env bash

python tools/visualize_result.py  --outdir vis_nyu_result_all \
--title_names RGB HHA GT Oracle SrcOnly_EarlyFusion Adapt_EarlyFusion Adapt_LateFusion Adapt_TripletaskHHA Adapt_TripletaskBoundary Adapt_TripletaskRefined  \
--indir_list \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/nyu-trainval_rgbhha_only_6ch---nyu-test_rgbhha/640x480-drn_d_38-100.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha_only_6ch---nyu-test_rgbhha/normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch_MFNet---nyu-test_rgbhha/MCD-MFNet-AddFusion-normal-drn_d_38-15/label \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/depth \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/boundary \
/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/refined_label
