# Multichannel Semantic Segmentation with Unsupervised Domain Adaptation implemeted by PyTorch

## Installation
Use **Python 2.x**

First, you need to install PyTorch following [the official site instruction](http://pytorch.org/).

Next, please install the required libraries as follows;
```
pip install -r requirements.txt
```

## Usage
We adopted Maximum Classifier Discrepancy (MCD) for unsupervised domain adaptation.


### MCD Training

- adapt_xxx.py
    - for domain adaptation (MCD)
- dann_xxx.py
    -  for domain adaptation (DANN: Domain Adversarial Neural Network)  

- source_xxx.py
    - for source only
    


#### Fusions
Early Fusion
```
python adapt_trainer.py suncg nyu --input_ch 6 --src_split train_rgbhhab --tgt_split trainval_rgbhha
```

Late Fusion
```
python adapt_mfnet_trainer.py suncg nyu --input_ch 6 --src_split train_rgbhhab --tgt_split trainval_rgbhha --method_detail MFNet-AddFusion
```
Score Fusion
```
python adapt_mfnet_trainer.py suncg nyu --input_ch 6 --src_split train_rgbhhab --tgt_split trainval_rgbhha --method_detail MFNet-ScoreAddFusion
```

#### Multitask

Segmentation + Depth Estimation (HHA regression)
```
python adapt_multitask_trainer.py suncg nyu --input_ch 6 --src_split train_rgbhhab --tgt_split trainval_rgbhha --method_detail MFNet-ScoreAddFusion
```


Segmentation + Depth Estimation (HHA regression) + Boundary Detection
```
python adapt_tripletask_trainer.py suncg nyu --input_ch 6 --src_split train_rgbhhab --tgt_split trainval_rgbhha  --method_detail MFNet-ScoreAddFusion
```


### Test
```
python adapt_triple_multitask_tester.py nyu --split test_rgbhha train_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask/pth/MCD-normal-drn_d_38-10.pth.tar
```

Results will be saved under "./test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/MCD-normal-drn_d_38-10.tar/" .

### Postprocess using Boundary Detection output
You need Matlab.

```
bash ./sample_scripts/refine_seg_by_boundary.sh
```


### Evaluation for Validation Data

```
python eval.py nyu ./test_output/suncg-train_rgbhhab2nyu-trainval_rgbhha_6ch_MCD_triple_multitask---nyu-test_rgbhha/YOUR_MODEL_NAME/label
```


## Reference codes
- https://github.com/Lextal/pspnet-pytorch
- https://github.com/fyu/drn
- https://github.com/meetshah1995/pytorch-semseg
- https://github.com/ycszen/pytorch-seg
