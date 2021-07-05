#!/bin/bash

# The command lacks the weights array file for Grid search and Diff_evolution
# The command lacks the historiesFolder_validationErrorInverse


if [ $1 = "Global_evaluate_models" ]; then
    python evaluate_ensemble.py -op $1 \
                                -mlist $2 \
                                -fn $3 \
                                -rf "Results" \
                                -af 3 \
                                -is "test" \
                                -tmf "Trained_models/" \
                                -cs "unbalanced"

elif [ $1 = "Combine_ensembles" ]; then
	python evaluate_ensemble.py -op $1 \
                                -mlist $2 \
                                -fn $3 \
                                -rf "Results" \
                                -af 3 \
                                -is "test" \
                                -tmf "Trained_models/" \
                                -cs "unbalanced"
elif [ $1 = "Confusion_matrices" ] || [ $1 = "Difference_matrices" ]; then
    if [ $2 = "Global" ]; then 
        python evaluate_ensemble.py -op $1 \
                                    -et $2 \
                                    -mlist $3 \
                                    -fn $4 \
                                    -rf "Results" \
                                    -af 3 \
                                    -is "test" \
                                    -tmf "Trained_models/" \
                                    -cs "unbalanced"
    else
        python evaluate_ensemble.py -op $1 \
                                    -et $2 \
                                    -mt $3 \
                                    -tc $4 \
                                    -wt $5 \
                                    -b $6 \
                                    -w $7 \
                                    -ofs $8 \
                                    -as $9 \
                                    -fn ${10} \
                                    -af 3 \
                                    -is "test" \
                                    -tmf "Trained_models/" \
                                    -rf "Results" \
                                    -cs "unbalanced" \
                                    -hf_vei "Data/Weights/"
    fi
else
    python evaluate_ensemble.py -op $1 \
                                -mt $2 \
                                -tc $3 \
                                -wt $4 \
                                -b $5 \
                                -w $6 \
                                -ofs $7 \
                                -as $8 \
                                -fn $9 \
                                -af 3 \
                                -is ${10} \
                                -tmf "Trained_models/" \
                                -rf "Results" \
                                -cs "unbalanced" \
                                -hf_vei "Data/Weights/"
fi
