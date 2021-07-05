#!/bin/bash

echo "Insert the operation name : 
    ['Confusion_matrices', 
    'Difference_matrices', 
    'Evaluate_ensembles', 
    'Store_models_probabilities', 
    'StickDiagrams_wellClassifiedClips_per_numberOfModels',
    'Global_evaluate_models',
    'Combine_ensembles']"
read operation

if [ $operation = "Global_evaluate_models" ] ; then
    echo "Insert the number of folds"
    read folds_number
    echo "Would like to mention the models to integrate in the global ensemble ? [Yes/No]"
    read integrate_models
    echo "Which sets are invovled ? [test/train_val]"
    read involved_sets
    if [ $integrate_models = "Yes" ] ; then
        echo "What is the list of models : Example TWOSTREAM_I3D_PRETRAINED"
        read models_list
    else
        declare -a models_list=("SPECIALCASE_PRETRAINED R3D_34_SCRATCH TWOSTREAM_I3D_PRETRAINED TWOSTREAM_I3D_SCRATCH C3D_PRETRAINED C3D_SCRATCH I3D_PRETRAINED I3D_SCRATCH")
    fi

    sbatch evaluate_ensemble.sh $operation "${models_list[@]}" $folds_number $involved_sets

elif [ $operation = "Combine_ensembles" ] ; then
    echo "Insert the number of folds"
    read folds_number
    echo "Would you like to mention the models to integrate in the combination of global ensembles ? [Yes/No]"
    read integrate_models
    echo "Which sets are invovled ? [test/train_val]"
    read involved_sets
    if [ $integrate_models = "Yes" ] ; then
        echo "What is the list of models : Example TWOSTREAM_I3D_PRETRAINED TWOSTREAM_I3D_SCRATCH"
        read models_list
    else
        declare -a models_list=("SPECIALCASE_PRETRAINED R3D_34_SCRATCH TWOSTREAM_I3D_PRETRAINED TWOSTREAM_I3D_SCRATCH C3D_PRETRAINED C3D_SCRATCH I3D_PRETRAINED I3D_SCRATCH")
    fi

    sbatch evaluate_ensemble.sh $operation "${models_list[@]}" $folds_number $involved_sets

elif [ $operation = "Confusion_matrices" ] || [ $operation = "Difference_matrices" ] ; then
    echo "Insert the ensemble type [Unique/Global]"
    read ensemble_type
    if [ $ensemble_type = "Global" ] ; then
        echo "Insert the number of folds"
        read folds_number
        echo "Would you like to mention the models to integrate in the computation of the confusion matrices for the global ensemble ? [Yes/No]"
        read integrate_models
        if [ $integrate_models = "Yes" ] ; then
            echo "What is the list of models : Example TWOSTREAM_I3D_PRETRAINED TWOSTREAM_I3D_SCRATCH"
            read models_list
        else
            declare -a models_list=("TWOSTREAM_I3D_PRETRAINED_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3 TWOSTREAM_I3D_PRETRAINED C3D_PRETRAINED I3D_SCRATCH")
        fi
        sbatch evaluate_ensemble.sh $operation $ensemble_type "${models_list[@]}" $folds_number 
    else
        echo "Choose any of the following model types : [TWOSTREAM_I3D,I3D,C3D,R3D_18,R3D_34,R3D_50,R3D_101,R3D_152]"
        read model_type
        echo "Choose any of the training preconditions : [_PRETRAINED,_SCRATCH]"
        read training_condition
        echo "Insert the augmentation status : ['non_augmented', 'augmented_onTheFly', 'augmented_precomputed']"
        read augmentation_status
        echo "Insert the optical flow status : ['TVL1_precomputed', 'FarneBack_onTheFly']"
        read optical_flow_status
        echo "Insert the weighting method type : ['GRID_SEARCH', 'DIFFERENTIAL_EVOLUTION', 'SUM', 'VALIDATION_ERROR_INVERSE', 'MAXIMUM']"
        read weights_type
        echo "Insert batch_size"
        read batch_size
        echo "Insert the number of workers"
        read workers
        echo "Insert the number of folds"
        read folds_number

        sbatch evaluate_ensemble.sh $operation $ensemble_type $model_type $training_condition $weights_type $batch_size $workers $optical_flow_status $augmentation_status $folds_number
    fi

else
    echo "Choose any of the following model types : [TWOSTREAM_I3D,I3D,C3D,R3D_18,R3D_34,R3D_50,R3D_101,R3D_152]"
    read model_type
    echo "Choose any of the training preconditions : [_PRETRAINED,_SCRATCH]"
    read training_condition
    echo "Insert the augmentation status : ['non_augmented', 'augmented_onTheFly', 'augmented_precomputed']"
    read augmentation_status
    echo "Insert the optical flow status : ['TVL1_precomputed', 'FarneBack_onTheFly']"
    read optical_flow_status
    echo "Insert the weighting method type : ['GRID_SEARCH', 'DIFFERENTIAL_EVOLUTION', 'SUM', 'VALIDATION_ERROR_INVERSE', 'MAXIMUM']"
    read weights_type
    echo "Insert batch_size"
    read batch_size
    echo "Insert the number of workers"
    read workers
    echo "Insert the number of folds"
    read folds_number
    echo "Which sets are invovled ? [test/train_val]"
    read involved_sets

    sbatch evaluate_ensemble.sh $operation $model_type $training_condition $weights_type $batch_size $workers $optical_flow_status $augmentation_status $folds_number $involved_sets
fi

