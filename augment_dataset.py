import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import os
import re
import csv
import argparse
from vidaug import augmentors as va
import cv2

##################################################################
# Utils functions
##################################################################

def augmentor(frame_shape, augmentation_probability):
    """
    Prepares the video data augmentator by applying some augmentations to it
    """
    height = frame_shape[0]
    width = frame_shape[1]
    sometimes = lambda aug: va.Sometimes(augmentation_probability, aug)  # Used to apply augmentor with 75% probability

    seq = va.Sequential([
        # randomly crop video with a size of (height-60 x width-60)
        # height and width are in this order because the instance is a ndarray
        sometimes(va.RandomCrop(size=(height - 60, width - 60))),
        sometimes(va.HorizontalFlip()),  # horizontally flip
        sometimes(va.Salt(ratio=100)),  # salt
        sometimes(va.Pepper(ratio=100))  # pepper
    ])
    return seq

def write_video(video_path, video_images):
    """
    Outputs a video file from a list of video_images
    :param video_path : path to the written video
    :param video_images : frames list of the video
    """
    # Define the video shape
    width = video_images[0].shape[0]
    height = video_images[0].shape[1]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    # Write out frame to video
    for image in video_images:
        out.write(image)
    # Release everything if job is finished
    out.release()

def augment_video(video_path):
    """
    Augment an rgb video clip
    :param video_path : path to the rgb video clip
    return a list of augmented frames
    """
    cap = cv2.VideoCapture(video_path)

    # Extract video frames
    frames = list()
    if not cap.isOpened():
        cap.open(video_path)
    ret = True
    while (True and ret):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # Augment video frames
    augmentation_probability = 0.85
    seq = augmentor(frames[0].shape, augmentation_probability)
    augmented_frames = seq(frames)

    # Resizing frames to fit the decided setup
    frame_height = frame_width = 224
    resized_augmented_frames = list()
    for augmented_frame in augmented_frames:
        resized_augmented_frame = cv2.resize(augmented_frame, (frame_height, frame_width))
        resized_augmented_frames.append(resized_augmented_frame)

    return np.asarray(resized_augmented_frames)


def augment_folds(augmented_data_subfolder_path, folds_subfolder_path, augmentation_frequency, operation):
    """
    Function used to augment data
    :param dataset_directory : path to the video dataset to augment
    :param folds_subfolder_path : name of the 
    """

    # Getting arguments
    nb_folds = int(os.path.basename(folds_subfolder_path)[0])
    # Determine the list of fold_indexes
    fold_indexes = list(range(0, nb_folds))
    # Add the videos to the dataset
    for fold_index in fold_indexes:
        foldfile = "fold"+str(fold_index)+".csv"
        foldfile_path = os.path.join(folds_subfolder_path, foldfile)
        fold_dataframe = pd.read_csv(foldfile_path)
        # Get the fold videos list
        fold_videos = fold_dataframe['rgbclips_path'].values
        # Augment the videos as much frequently as asked for (augmentation_frequency)
        for frequency_index in range(0, augmentation_frequency):
            currentAddedVideos_list = list()
            column_name = "rgbclips_augmented_"+str(frequency_index)+"_path"
            if column_name not in fold_dataframe.columns:
                for video_path in fold_videos:
                    # Add the links to the csv file
                    video_extension = "_augmented_"+str(frequency_index) + ".mp4"
                    augmented_video_path = os.path.join(augmented_data_subfolder_path, os.path.splitext(os.path.basename(video_path))[0] + video_extension)
                    currentAddedVideos_list.append(augmented_video_path)
                    # Create the video within the augmentation subfolder if the operation is set to augment_videos
                    # Otherwise, we only need to update the links of the augmented videos
                    if operation == "augment_videos":
                        augmented_video = augment_video(video_path)
                        write_video(augmented_video_path, augmented_video)
                        print(augmented_video_path)
                fold_dataframe[column_name] = currentAddedVideos_list
        fold_dataframe.to_csv(foldfile_path, index=False)
                    


##################################################################
# Main program : augment dataset
##################################################################

def augment_dataset(dataset_directory, folds_subfolder_path, augmentation_frequency):
    """
    Function used to augment data
    :param dataset_directory : path to the video dataset to augment
    :param folds_subfolder_path : name of the 
    """
    # Create the augmented video data subfolder
    augmented_data_subfolder = "augmented_frequency_" + str(augmentation_frequency)
    augmented_data_subfolder_path = os.path.join(dataset_directory, augmented_data_subfolder)
    if not os.path.exists(augmented_data_subfolder_path):
        operation = "augment_videos"
        os.mkdir(augmented_data_subfolder_path)
        print("Augment video data %d times."%(augmentation_frequency))
        augment_folds(augmented_data_subfolder_path, folds_subfolder_path, augmentation_frequency, operation)
    else:
        operation = "update_augmentedData_links"
        print("Video data was already augmented %d times."%(augmentation_frequency))
        augment_folds(augmented_data_subfolder_path, folds_subfolder_path, augmentation_frequency, operation)


def main(args):
    augment_dataset(args.dataset_directory, args.folds_subfolder_path, args.augmentation_frequency)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-df',
                        '--dataset_directory',
                        help='Specify the path to the data folder',
                        type=str,
                        default="Data/Crowd-11",
                        required=True)
    parser.add_argument('-fsp',
                        '--folds_subfolder_path',
                        help='Specify the path to the folds subfolder',
                        type=str,
                        default="Folds/5_folds",
                        required=True)
    parser.add_argument('-af',
                        '--augmentation_frequency',
                        help='Specify the frequency to which we augment the dataset.',
                        type=int,
                        default=5,
                        required=True)
    args = parser.parse_args()
    main(args)