#!/bin/bash

python -u train.py \
-trp $1 \
-vp $2 \
-tsp $3 \
-mt $4 \
-tc $5 \
-fn $6 \
-b $7 \
-w $8 \
-cs $9 \
-as ${10} \
-af ${11} \
-ofs ${12} \
-emwf "Data/Weights" \
-tmf ${13} \
-e ${14}