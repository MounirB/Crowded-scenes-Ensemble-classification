#!/bin/bash

# We use this script if any job crashes due to memory-related problems

python -u train.py \
-trp "Folds/5_folds/split_test0_val1/train.csv" \
-vp "Folds/5_folds/split_test0_val1/val.csv" \
-tsp "Folds/5_folds/split_test0_val1/test.csv" \
-mt "TWOSTREAM_I3D" \
-tc "_SCRATCH" \
-fn 5 \
-b 8 \
-w 1 \
-cs "unbalanced" \
-as "non_augmented" \
-af 0 \
-ofs "TVL1_precomputed" \
-emwf "Data/Weights/" \
-tmf "Trained_models/" \
-e 30