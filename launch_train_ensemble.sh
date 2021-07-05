#!/bin/bash

echo "Choose any of the following model types : [TWOSTREAM_I3D,I3D,C3D,R3D_18,R3D_34,R3D_50,R3D_101,R3D_152]"
read model_type
echo "Choose any of the training preconditions : [_PRETRAINED,_SCRATCH]"
read training_condition
echo "Choose the augmentation status : [non_augmented, augmented_onTheFly, augmented_precomputed]"
read augmentation_status

if [ $augmentation_status = "augmented_onTheFly" ] || [ $augmentation_status = "augmented_precomputed" ] ; then
	echo "What is the augmentation frequency ?"
	read augmentation_frequency
else
	augmentation_frequency=0
fi

echo "Choose the optical flow status : [TVL1_precomputed, FarneBack_onTheFly]"
read optical_flow_status
echo "Write the number of folds"
read folds_number
echo "Insert batch_size"
read batch_size
echo "Insert the number of workers"
read workers
echo "Insert the number of epochs"
read epochs

python launch_train_ensemble.py \
-fn $folds_number \
-tmf "Trained_models/" \
-mt $model_type \
-tc $training_condition \
-as $augmentation_status \
-af $augmentation_frequency \
-ofs $optical_flow_status \
-df "Data/Crowd-11/" \
-cs "unbalanced" \
-pff "Folds/" \
-db "Data/database.csv" \
-b $batch_size \
-w $workers \
-e $epochs