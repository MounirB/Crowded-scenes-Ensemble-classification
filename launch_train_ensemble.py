import os
import pandas as pd
from generate_folds import generate_folds
import argparse
import subprocess
import glob
import shutil
from augment_dataset import augment_dataset

##################################################################
# Utils functions
##################################################################

def sortOut_future_trainedModels(folds_subfolder_path,
                                folds_number, 
                                trained_models_folder, 
                                model_type, 
                                training_condition,
                                optical_flow_status,
                                augmentation_status,
                                augmentation_frequency,
                                classes_status):
    """
    Classify the trained models into the folder which name describes the training settings
    - Within this folder each models category which constitues an ensemble is classified within a TestSplit subfolder
    - Within a TestSplit subfolder must be stored a test csv file as well as a train and a val csv files snippets
    - These train and val csv files snippets will be used for the weights retrieval procedures (i.e. grid search, differential evolution)
    - We only need a train and a validation files of a unique split of an Ensemble
    """
    ## Generate trained_models_subfolder within trained_models_folder
    if augmentation_status == "augmented_precomputed":
        trained_models_subfolder_name = str(folds_number) + \
	        "folds_" + \
	        model_type + \
	        training_condition + \
            "_CS_" + \
            classes_status + \
	        "_OF_" + \
	        optical_flow_status + \
	        "_AS_" + \
	        augmentation_status + \
            "_Freq"+str(args.augmentation_frequency)
    else:
        trained_models_subfolder_name = str(folds_number) + \
	        "folds_" + \
	        model_type + \
	        training_condition + \
            "_CS_" + \
            classes_status + \
	        "_OF_" + \
	        optical_flow_status + \
	        "_AS_" + \
	        augmentation_status

    if not os.path.exists(args.trained_models_folder):
            os.mkdir(args.trained_models_folder)
    trained_models_subfolder_path = os.path.join(trained_models_folder, trained_models_subfolder_name)
    if not os.path.exists(trained_models_subfolder_path):
        os.mkdir(trained_models_subfolder_path)
    ## Create TestSplit subfolders within trained models subfolder
    for test_index in range(0, folds_number):
        ## Create the TestSplit subfolder
        testsplit_name = "TestSplit" + str(test_index)
        testsplit_path = os.path.join(trained_models_subfolder_path, testsplit_name)
        if not os.path.exists(testsplit_path):
            os.mkdir(testsplit_path)
        
        ## Move a snippet of the train/val/test csv files within the TestSplit
        # Determine the validation indexes using the folds_number and excluding the test_index
        val_folds_indices = list(range(0,folds_number))
        val_folds_indices.remove(test_index)
        # Only one split from the test/train/val combination suffices
        val_index = val_folds_indices[0]
        # Determine the split specification name
        split_specification = "split_test"+str(test_index)+"_val"+str(val_index)
        # Get the split subfolder contents (within Folds folder)
        folds_split_subfolder_path = os.path.join(folds_subfolder_path, split_specification)
        folds_split_subfolder_content = glob.glob(os.path.join(folds_split_subfolder_path, "*.csv"))
        # Copy a representative split subfolder contents to TestSplit current subfolder
        for csvfile in folds_split_subfolder_content:
            if not os.path.exists(os.path.join(testsplit_path, os.path.basename(csvfile))):
                shutil.copyfile(csvfile, os.path.join(testsplit_path, os.path.basename(csvfile)))

##################################################################
# Launch training ensembles
##################################################################

def launcher(args):
    # Get arguments
    folds_subfolder = str(args.folds_number) + "_folds"
    folds_subfolder_path = os.path.join(args.parent_folds_folder, folds_subfolder)
    test_folds_indexes = list(range(0, args.folds_number))

    # Generate folds and folders if they don't already exist
    if not os.path.exists(folds_subfolder_path):
        generate_folds(args)

    # Sort out the future trained model files within the folds subfolders 
    sortOut_future_trainedModels(folds_subfolder_path,
                                args.folds_number, 
                                args.trained_models_folder, 
                                args.model_type, 
                                args.training_condition,
                                args.optical_flow_status,
                                args.augmentation_status,
                                args.augmentation_frequency,
                                args.classes_status)

    # Pre_augment the data if required by the user or update the augmented data links
    if args.augmentation_status == "augmented_precomputed":
        augment_dataset(args.dataset_directory, folds_subfolder_path, args.augmentation_frequency)

    ## We test different test/val folds combinations
    # Create a process for each training procedure
    list_processes = list()
    # Determination of the test index
    for test_index in test_folds_indexes:
        test_set = pd.read_csv(os.path.join(folds_subfolder_path, "fold"+str(test_index)+".csv"), index_col=0)
        # Create the possible validation folds to select by excluding the selected test index
        val_folds_indexes = test_folds_indexes.copy()
        val_folds_indexes.remove(test_index)
        # Determination of the validation index
        for val_index in val_folds_indexes:
            val_set = pd.read_csv(os.path.join(folds_subfolder_path, "fold" + str(val_index) + ".csv"), index_col=0)
            train_set_list = [os.path.join(folds_subfolder_path, "fold" + str(trainfold_index) + ".csv") for trainfold_index in
                              val_folds_indexes if trainfold_index != val_index]
            train_set = pd.concat([pd.read_csv(train_fold, index_col=0) for train_fold in train_set_list])
            split_folder = os.path.join(folds_subfolder_path, "split"+"_test"+str(test_index)+"_val"+str(val_index))
            # Create the split subfolder
            if not os.path.exists(split_folder):
                os.mkdir(split_folder)
            # Create the links for the splits csv files
            val_path = os.path.join(split_folder, "val.csv")
            train_path = os.path.join(split_folder, "train.csv")
            test_path = os.path.join(split_folder, "test.csv")
            # Creat the csv files for each split
            if not os.path.exists(val_path):
                val_set.to_csv(val_path)
            if not os.path.exists(train_path):
                train_set.to_csv(train_path)
            if not os.path.exists(test_path):
                test_set.to_csv(test_path)
            # Launch a local model training
            os.system("sbatch train.sh "\
                +train_path+\
                " "+val_path+\
                " "+test_path+\
                " "+args.model_type+\
                " "+args.training_condition+\
                " "+str(args.folds_number)+\
                " "+str(args.batch_size)+\
                " "+str(args.workers)+\
                " "+args.classes_status+\
                " "+args.augmentation_status+\
                " "+str(args.augmentation_frequency)+\
                " "+args.optical_flow_status+\
                " "+args.trained_models_folder+\
                " "+str(args.epochs))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn',
                        '--folds_number',
                        help='Specify the number of folds.',
                        type=int,
                        default = 5,
                        required=True)
    parser.add_argument('-tmf',
                        '--trained_models_folder',
                        help='Specify the path to the trained models.',
                        type=str,
                        default='Trained_models/',
                        required=True)
    parser.add_argument('-mt',
                        '--model_type',
                        help='Specify the model name.',
                        type=str,
                        choices=['TWOSTREAM_I3D', 'I3D', 'C3D', 'R3D_18', 'R3D_34', 'R3D_50', 'R3D_101', 'R3D_152'],
                        required=True)
    parser.add_argument('-tc',
                        '--training_condition',
                        help='Specify how was the weights'' state of the model before going to be trained on Crowd-11.',
                        type=str,
                        choices=['_SCRATCH', '_PRETRAINED'],
                        required=True)
    parser.add_argument('-as',
                        '--augmentation_status',
                        help='Mentions if we want to apply or not data augmentation.',
                        choices=['non_augmented', 'augmented_onTheFly', 'augmented_precomputed'],
                        type=str,
                        default='non_augmented',
                        required=True)
    parser.add_argument('-af',
                        '--augmentation_frequency',
                        help='Specify the frequency to which we augment the dataset.',
                        type=int,
                        default=5,
                        required=True)
    parser.add_argument('-ofs',
                        '--optical_flow_status',
                        help='Specifies if the optical flow was pre-computed (happens with TV-L1) or is computed on-the-fly (happens with Farneback).',
                        type=str,
                        choices=['TVL1_precomputed', 'FarneBack_onTheFly'],
                        required=True)
    parser.add_argument('-df',
                        '--dataset_directory',
                        help='Specify the path to the data folder',
                        type=str,
                        default="Data/Crowd-11/",
                        required=True)
    parser.add_argument('-cs',
                        '--classes_status',
                        help='Mentions if we want to make the data balanced or keep it as is.',
                        choices=['balanced', 'unbalanced'],
                        type=str,
                        default='unbalanced',
                        required=True)
    parser.add_argument('-pff',
                        '--parent_folds_folder',
                        help='Specify the path to the folds folder',
                        type=str,
                        default="Folds/",
                        required=True)
    parser.add_argument('-db',
                        '--database_file',
                        help='Specify the path to the database file. The database file is the original file describing the distribution of the data into scenes/videos/clips that can allow to split the dataset into folds',
                        type=str,
                        default="Data/database.csv",
                        required=True)
    parser.add_argument('-b',
                        '--batch_size',
                        help='Specify the batch_size for training.',
                        type=int,
                        required=True)
    parser.add_argument('-w',
                        '--workers',
                        help='Specify the number of GPUs involved in the computation.',
                        type=int,
                        default = 1,
                        required=True)
    parser.add_argument('-e',
                        '--epochs',
                        help='Specify the number of epochs for training.',
                        type=int,
                        required=True)

    args = parser.parse_args()
    launcher(args)