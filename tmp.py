tgt_dataset = "nyu"
label_outdir = "test_output/suncg-train_rgbhha2nyu-all_rgbhha_6ch_Finetune_MFNet---nyu-test_rgbhha/MCD-MFNet-ScoreConcatConvFusion-MCD-normal-drn_d_38-10ANDMCD-normal-drn_d_38-10-drn_d_38-5/label"

eval_str = "python eval.py %s %s" % (tgt_dataset, label_outdir)
print (eval_str)
import subprocess

subprocess.call(eval_str, shell=True)
