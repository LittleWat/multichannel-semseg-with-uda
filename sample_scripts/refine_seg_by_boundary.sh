#!/usr/bin/env bash

#BOUNDARY_DIR="/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar/boundary"
#BASE_DIR="/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-20.tar"

BOUNDARY_DIR="/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/edges/canny"
BASE_DIR="/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/test_output/gta-train2city-train_3ch---city-val/MCD-numk2-drn_d_105-10.tar/label"


SEG_DIR="${BASE_DIR}/label"
BWBD_DIR="${BASE_DIR}/bwboundary"

BINARY_BOUNDARY_DIR="${BOUNDARY_DIR}_binary"

python tools/binalize_boundary.py ${BOUNDARY_DIR}
matlab -nodesktop -nosplash -r "addpath('./tools'); apply_bwboundary('${BINARY_BOUNDARY_DIR}'); exit"
python tools/refine_seg_by_bwboundary.py ${SEG_DIR} ${BWBD_DIR}
