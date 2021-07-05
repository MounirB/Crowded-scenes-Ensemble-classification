import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import os
import re
import csv
import argparse

##################################################################
# Utils functions
##################################################################

def verify_folds_intersection(folds_list):
    """
    Verify that the folds scenes do not overlap
    """
    for num_fold in range(0, len(folds_list)):
        selected_fold_scenes = folds_list[num_fold]
        for num_concurrent_fold in range(0, len(folds_list)):
            if num_fold == num_concurrent_fold:
                continue
            else:
                print(num_fold, " ", num_concurrent_fold, " ", set(folds_list[num_fold]).intersection(set(folds_list[num_concurrent_fold])))

def sum_folds_lengths(folds_list):
    """
    Verify that we retrieve the number of scenes via the sum of the lengths of the folds
    """
    summation = 0
    for num_fold in range(0, len(folds_list)):
        summation = summation + len(folds_list[num_fold])
    return summation

def each_fold_length(folds_list):
    """
    Display each fold's length
    """
    for num_fold in range(0, len(folds_list)):
        print(len(folds_list[num_fold]))

def make_folds_csvs(dataset_folder, parent_folds_folder, database, folds_scenes, nb_folds):
    """
    Create the csv file containing the links to each version of a clip (rgb/flow) for each fold
    """
    rgb_dataset_directory = os.path.join(dataset_folder, "rgb")
    flow_dataset_directory = os.path.join(dataset_folder, "flow")

    # Creation of the folds folder and subfolder
    subfolder = str(nb_folds) + "_folds"
    folds_folder = os.path.join(parent_folds_folder, subfolder)
    if not os.path.exists(folds_folder):
        os.makedirs(folds_folder)

    # Get the list of videos
    videos = os.listdir(rgb_dataset_directory)
    flowvideos = [os.path.splitext(os.path.basename(video))[0] +"_x.avi" for video in videos] + \
     [os.path.splitext(os.path.basename(video))[0] +"_y.avi" for video in videos]

    # Splitting the dataset
    labels = [re.findall("^[0-9][0-9]?", video)[0] for video in videos]
    # fold csv
    for iteration in range(0, nb_folds):
        folds_database = database.loc[database['scene_number'].isin(folds_scenes[iteration])]
        fold_video_names = [re.findall("(.*)\.[ma][pv][4i]", video_name)[0] for video_name in
                         folds_database['video_name'].values]
        fold_rgbvideos = [video for video in videos if
                        re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)\.[ma][pv][4i]", video)[0] in fold_video_names]

        fold_yflowvideos, fold_xflowvideos = list(), list()
        for video in flowvideos:
            if re.findall('.*_x\.[ma][pv][i4]', video):
                fold_xflowvideos.append(video)
        for video in flowvideos:
            if re.findall('.*_y\.[ma][pv][i4]', video):
                fold_yflowvideos.append(video)

        fold_rgbvideos = sorted(fold_rgbvideos)
        fold_yflowvideos = [video for video in fold_yflowvideos if
                             re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_y\.[ma][pv][4i]", video)[0] in fold_video_names]
        fold_yflowvideos = sorted(fold_yflowvideos)
        fold_xflowvideos = [video for video in fold_xflowvideos if
                             re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_x\.[ma][pv][4i]", video)[0] in fold_video_names]
        fold_xflowvideos = sorted(fold_xflowvideos)
        

        fold_labels = [re.findall("^[0-9][0-9]?", video)[0] for video in fold_rgbvideos]
        fold = [[os.path.join(rgb_dataset_directory, rgbvideo), os.path.join(flow_dataset_directory, xflowvideo),
                  os.path.join(flow_dataset_directory, yflowvideo), label] for rgbvideo, xflowvideo, yflowvideo, label in
                 zip(fold_rgbvideos, fold_xflowvideos, fold_yflowvideos, fold_labels)]

        # create_csvs
        fold_file = 'fold'+ str(iteration) +'.csv'
        with open(os.path.join(folds_folder, fold_file), 'w', newline='') as csvfile:
            mywriter = csv.writer(csvfile, delimiter=',')
            mywriter.writerow(['rgbclips_path', 'x_axis_flowclips_path', 'y_axis_flowclips_path', 'class'])
            for video in fold:
                mywriter.writerow(video)
            print('Fold '+str(iteration)+' CSV file created successfully')

def folds_histograms(dataset_folder, database, folds_list):
    """
    Displays the histogram of each fold.
    Shows the frequency of each label for each fold.
    """
    videos = os.listdir(dataset_folder)
    for fold_scenes in folds_list:
        fold_database = database.loc[database['scene_number'].isin(fold_scenes)]
        fold_videos = [video for video in videos if
                       re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*?\.[ma][pv][4i])", video)[0] in fold_database[
                           'video_name'].values]
        fold_labels = [int(re.findall("^[0-9][0-9]?", video)[0]) for video in fold_videos]
        plt.hist(fold_labels, bins=11)
        plt.show()

def scenes_counter(all_scenes_set, all_scenes_list):
    """
    Counting the number of sequences per scene
    """
    all_scenes_count = list()
    for scene in list(all_scenes_set):
        all_scenes_count.append(all_scenes_list.count(scene))
    return all_scenes_count

def classes_counter(database, nb_classes):
    """
    Provides the sequences frequency for each class
    """
    classes = list(database['label'].values)
    ### Class occurrences counter
    frequences = [classes.count(int(label)) for label in range(0, nb_classes)]
    return frequences

def determine_min_score_fold_index(folds_distribs, nb_classes):
    """
    Returns the index of the fold with the smallest distribution score
    """
    folds_scores = [sum(distrib)/nb_classes for distrib in folds_distribs]
    min_score_index = folds_scores.index(min(folds_scores))
    return min_score_index

def update_fold_distribution(dataset_directory, fold_distribution, num_scene, dataframe, frequences, nb_folds):
    """
    Updates the folds weights according to the number of labels each of their scenes possess.
    """
    rgb_dataset_directory = os.path.join(dataset_directory, "rgb")
    videos = os.listdir(rgb_dataset_directory)
    scene_database = dataframe.loc[dataframe['scene_number'] == num_scene]
    scene_videos = [video for video in videos if
                   re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*?\.[ma][pv][4i])", video)[0] in scene_database[
                       'video_name'].values]
    scene_labels = [int(re.findall("^[0-9][0-9]?", video)[0]) for video in scene_videos]
    for scene_label in scene_labels:
        fold_distribution[scene_label] = fold_distribution[scene_label] + 1/(frequences[scene_label]/nb_folds)

    return fold_distribution


##################################################################
# Main program : generate folds
##################################################################

def generate_folds(args):
    """
    Function used to generate folds. Calls other subfunctions
    """
    # Getting arguments
    database = pd.read_csv(args.database_file)
    nb_classes = len(set(database['label']))
    nb_folds = args.folds_number
    dataset_directory = args.dataset_directory

    # Number of existing scenes and list of sets of scenes
    all_scenes_set = list(set(database['scene_number'].values))
    all_scenes_list = list(database['scene_number'].values)
    nb_scenes = len(all_scenes_set)

    scenes_frequencies = scenes_counter(all_scenes_set, all_scenes_list)
    folds_distrib = np.zeros((nb_folds, nb_classes))
    total_classes_frequences = classes_counter(database, nb_classes)
    folds_scenes = list(np.zeros(nb_folds))

    for num_fold in range(0, nb_folds):
        folds_scenes[num_fold] = list()

    while all_scenes_set:
        fold_smallest_index = determine_min_score_fold_index(folds_distrib, nb_classes)
        biggest_scene_index = scenes_frequencies.index(max(scenes_frequencies))
        biggest_scene_number = all_scenes_set[biggest_scene_index]

        all_scenes_set.pop(biggest_scene_index)
        scenes_frequencies.pop(biggest_scene_index)

        folds_scenes[fold_smallest_index].append(biggest_scene_number)
        folds_distrib[fold_smallest_index] = update_fold_distribution(dataset_directory, folds_distrib[fold_smallest_index], biggest_scene_number, database, total_classes_frequences, nb_folds)

    ## Folds intersection verification
    # verify_folds_intersection(folds_scenes)
    
    ## Folds histograms
    # folds_histograms(database, folds_scenes)

    ## Create the csv file containing the links to each version of a clip (rgb/flow) for each fold
    make_folds_csvs(dataset_directory, args.parent_folds_folder, database, folds_scenes, nb_folds)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-df',
                        '--dataset_directory',
                        help='Specify the path to the data folder',
                        type=str,
                        default="Data/Crowd-11/",
                        required=True)
    parser.add_argument('-pff',
                        '--parent_folds_folder',
                        help='Specify the path to the folds folder',
                        type=str,
                        default="Folds/",
                        required=True)
    parser.add_argument('-db',
                        '--database_file',
                        help='Specify the path to the database file',
                        type=str,
                        default="Data/database.csv",
                        required=True)
    parser.add_argument('-fn',
                        '--folds_number',
                        help='Specify the number of folds.',
                        type=int,
                        default=5,
                        required=True)
    args = parser.parse_args()

    generate_folds(args)