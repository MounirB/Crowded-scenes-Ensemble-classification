from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

import warnings
import six
import os
import cv2
import sys
import numpy as np
import math
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
from keras.optimizers import Adam, SGD
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv3D, ZeroPadding3D, MaxPooling3D, AveragePooling3D, Dropout, Reshape, Lambda, Flatten
from keras.models import Model, Sequential
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from keras.layers.merge import add
from keras.regularizers import l2
from sklearn.utils import class_weight
import keras.backend as K
import keras
import traceback
import argparse
import random
from augment_dataset import augmentor
import types
import re
import skvideo.io

##################################################################
# Weights names for the Inception models
##################################################################

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

##################################################################
# Utils functions
##################################################################

def store_history(ensemble_models_weights_path,
                folds_number,
                split_specification,
                trained_model_name,
                history_):
    """
    Store the history containing the validation loss and accuracy of the best model
    from which a validation loss error inverse can be computed and used as a weight
    """
    # Create the weights folder if it doesn't exist
    test_index = re.findall(r'%s(\d+)' % "test", split_specification)[0]
    weights_subfolder = os.path.dirname(os.path.join(*(trained_model_name.split(os.path.sep)[1:])))
    weights_folder_path = os.path.join(ensemble_models_weights_path, weights_subfolder)
    if not os.path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)

    # Save the history of the model
    weights_file_name = os.path.basename(trained_model_name) + "_validation_losses.npy"
    weights_file_path = os.path.join(weights_folder_path, weights_file_name)
    np.save(weights_file_path, history_.history['val_loss'])


def str2bool(v):
    """
    This function interacts with argparse.
    As the boolean is not understood by argparse, we replace the dtype by str2bool
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def augment_dataframe(train_dataframe, augmentation_frequency):
    """
    Create the augmented dataframe by merging the augmented data columns with the original rgb clips columns
    """

    augmented_rgbclips = list(train_dataframe["rgbclips_path"].values)
    labels = list(train_dataframe["class"].values)
    x_axis_flowclips = list(train_dataframe["x_axis_flowclips_path"].values)
    y_axis_flowclips = list(train_dataframe["y_axis_flowclips_path"].values)
    TVL1_flowclips_length = len(x_axis_flowclips)
    for augmentation_stage in range(0, augmentation_frequency):
        augmented_rgbclips = augmented_rgbclips + list(train_dataframe["rgbclips_augmented_"+str(augmentation_stage)+"_path"].values)
        labels = labels + list(train_dataframe["class"].values)
        x_axis_flowclips = x_axis_flowclips + list(train_dataframe["x_axis_flowclips_path"].values)
        y_axis_flowclips = y_axis_flowclips + list(train_dataframe["y_axis_flowclips_path"].values)

    augmented_rgbclips_dict = {
        "rgbclips_path" : augmented_rgbclips,
        "x_axis_flowclips_path" : x_axis_flowclips,
        "y_axis_flowclips_path" : y_axis_flowclips,
        "class" : labels
    }

    augmented_train_dataframe = pd.DataFrame(augmented_rgbclips_dict, 
        columns=["rgbclips_path", "x_axis_flowclips_path", "y_axis_flowclips_path", "class"])

    return augmented_train_dataframe


##################################################################
# Data loading functions
##################################################################

def select_frames(frames, frames_per_video):
    """
    Select a certain number of frames determined by the number (frames_per_video)
    :param frames: list of frames
    :param frames_per_video: number of frames to select
    :return: selection of frames
    """
    step = len(frames) // frames_per_video
    if step == 0:
        step = 1
    first_frames_selection = frames[::step]
    final_frames_selection = first_frames_selection[:frames_per_video]

    return final_frames_selection


def get_twostream_videoclip(rgb_videoclip, flow_videoclip_axes, frames_per_video, frame_height, frame_width,
                            optical_flow_status='FarneBack_onTheFly', augmentation_status='non_augmented'):
    """
    From an RGB video clip returns an array of frames whose length is indicated by frames_per_video
    :param rgb_videoclip: the source video clip in RGB
    :param frames_per_video: number of frames per video to select
    :param frame_height: frame height
    :param frame_width: frame width
    :param optical_flow_status: 'FarneBack_onTheFly' or 'TVL1_precomputed'
    :param augmentation_status: 'non_augmented' or 'augmented_onTheFly'
    :return: selected number of frames
    """
    cap = cv2.VideoCapture(rgb_videoclip)

    frames = list()
    if not cap.isOpened():
        cap.open(rgb_videoclip)
    ret = True
    while (True and ret):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # Whether to apply or not data augmentation
    if augmentation_status == 'augmented_onTheFly' and optical_flow_status == 'FarneBack_onTheFly':
        # Application of data augmentation
        augmentation_probability = 0.75
        seq = augmentor(frames[0].shape, augmentation_probability)
        augmented_frames = seq(frames)
        frames = augmented_frames
        # Application of optical flow extraction
        flowframes = opticalflow_FarneBack_extractor(frames)

    elif (
            augmentation_status == 'non_augmented' or augmentation_status == 'augmented_precomputed') and optical_flow_status == 'FarneBack_onTheFly':
        # Application of optical flow extraction
        flowframes = opticalflow_FarneBack_extractor(frames)
    elif augmentation_status == 'augmented_onTheFly' and optical_flow_status == 'TVL1_precomputed':
        # Here the augmentation and the extraction of the augmented video clips were done offline
        flowframes = list(opticalflow_TVL1_retriever(flow_videoclip_axes))
    else:  # non_augmented and TVL1_precomputed optical flow
        flowframes = list(opticalflow_TVL1_retriever(flow_videoclip_axes))

    # Whether the optical flow was TVL1_precomputed or is computed on-the-fly
    if optical_flow_status == 'TVL1_precomputed':  # TVL1_precomputed TV-L1 optical flow
        # The following operations are intended to select a precise number of frames
        # and to resize them according to the decided setup of frame_height/width
        selected_rgbframes = select_frames(frames, frames_per_video)
        selected_xflowframes = select_frames(flowframes[0], frames_per_video)
        selected_yflowframes = select_frames(flowframes[1], frames_per_video)

        # Resizing frames to fit the decided setup
        resized_selected_rgbframes = list()
        resized_selected_flowframes = list()
        resized_selected_xflowframes = list()
        resized_selected_yflowframes = list()
        for selected_rgbframe, selected_xflowframe, selected_yflowframe in zip(selected_rgbframes, selected_xflowframes,
                                                                               selected_yflowframes):
            resized_selected_rgbframe = cv2.resize(selected_rgbframe, (frame_width, frame_height))
            resized_selected_rgbframes.append(resized_selected_rgbframe)
            resized_selected_xflowframe = cv2.resize(selected_xflowframe, (frame_width, frame_height))
            resized_selected_xflowframes.append(resized_selected_xflowframe)
            resized_selected_yflowframe = cv2.resize(selected_yflowframe, (frame_width, frame_height))
            resized_selected_yflowframes.append(resized_selected_yflowframe)

        resized_selected_rgbframes = np.asarray(resized_selected_rgbframes)
        resized_selected_xflowframes = np.asarray(resized_selected_xflowframes)
        resized_selected_xflowframes = np.expand_dims(resized_selected_xflowframes, axis=3)
        resized_selected_yflowframes = np.asarray(resized_selected_yflowframes)
        resized_selected_yflowframes = np.expand_dims(resized_selected_yflowframes, axis=3)
        resized_selected_flowframes = np.append(resized_selected_xflowframes, resized_selected_yflowframes, axis=3)

    else:  # on-the-fly FarneBack computed optical flow
        # The following operations are intended to select a precise number of frames
        # and to resize them according to the decided setup of frame_height/width
        selected_rgbframes = select_frames(frames, frames_per_video)
        selected_flowframes = select_frames(flowframes, frames_per_video)

        # Resizing frames to fit the decided setup
        resized_selected_rgbframes = list()
        resized_selected_flowframes = list()
        for selected_rgbframe, selected_flowframe in zip(selected_rgbframes, selected_flowframes):
            resized_selected_rgbframe = cv2.resize(selected_rgbframe, (frame_width, frame_height))
            resized_selected_rgbframes.append(resized_selected_rgbframe)
            resized_selected_flowframe = cv2.resize(selected_flowframe, (frame_width, frame_height))
            resized_selected_flowframes.append(resized_selected_flowframe)

        resized_selected_rgbframes = np.asarray(resized_selected_rgbframes)
        resized_selected_flowframes = np.asarray(resized_selected_flowframes)

    # return frame, video_clip
    return resized_selected_rgbframes, resized_selected_flowframes


def get_onestream_videoclip(rgb_videoclip, frames_per_video, frame_height, frame_width,
                            augmentation_status='non_augmented'):
    """
    From an RGB video clip returns an array of frames whose length is indicated by frames_per_video
    :param rgb_videoclip: the source video clip in RGB
    :param frames_per_video: number of frames per video to select
    :param frame_height: frame height
    :param frame_width: frame width
    :param optical_flow_status: 'FarneBack_onTheFly' or 'TVL1_precomputed'
    :param augmentation_status: 'non_augmented' or 'augmented_onTheFly'
    :return: selected number of frames
    """
    cap = cv2.VideoCapture(rgb_videoclip)

    frames = list()
    if not cap.isOpened():
        cap.open(rgb_videoclip)
    ret = True
    while (True and ret):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # Whether to apply or not data augmentation
    if augmentation_status == 'augmented_onTheFly':
        # Application of data augmentation
        augmentation_probability = 0.75
        seq = augmentor(frames[0].shape, augmentation_probability)
        augmented_frames = seq(frames)
        frames = augmented_frames

    # The following operations are intended to select a precise number of frames
    # and to resize them according to the decided setup of frame_height/width
    selected_rgbframes = select_frames(frames, frames_per_video)

    # Resizing frames to fit the decided setup
    resized_selected_rgbframes = list()
    for selected_rgbframe in selected_rgbframes:
        resized_selected_rgbframe = cv2.resize(selected_rgbframe, (frame_width, frame_height))
        resized_selected_rgbframes.append(resized_selected_rgbframe)

    resized_selected_rgbframes = np.asarray(resized_selected_rgbframes)
    # return video_clip frames
    return resized_selected_rgbframes


def opticalflow_FarneBack_extractor(frames):
    """
    Extract the optical flow from the list of frames that compose a clip
    :param frames: list of frames of a rgb clip
    :return: list of flows corresponding to the rgb clip
    """
    # Read first frame
    first_frame = frames[0]
    nb_frames = len(frames)
    # Scale and resize image
    resize_dim = 224
    max_dim = max(first_frame.shape)
    scale = resize_dim / max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    # Convert to gray scale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Store optical flow in flow_frames
    flow_frames = list()

    for numframe in range(1, nb_frames):
        if frames[numframe] is not None:
            # Convert new frame format`s to gray scale and resize gray frame obtained
            gray = cv2.cvtColor(frames[numframe], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # Calculate dense optical flow by Farneback method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=5, winsize=11,
                                                iterations=5,
                                                poly_n=5, poly_sigma=1.1, flags=0)
            flow_frames.append(flow)
            # Update previous frame
            prev_gray = gray

    # Convert flows list to np array
    flow_frames = np.asarray(flow_frames)

    return flow_frames


def opticalflow_TVL1_retriever(flow_videoclip_axes):
    """
    Get the pre-computed TV-L1 optical flow version of the RGB clip
    :param flow_videoclip_axes : list of the links to both of the flow axes
    """

    for flow_videoclip_axis in flow_videoclip_axes:
        capture = cv2.VideoCapture(flow_videoclip_axis)

        # Extract flow frames
        flow_frames = list()
        if not capture.isOpened():
            capture.open(flow_videoclip_axis)
        ret = True
        while (True and ret):
            ret, three_channeled_flow_frame = capture.read()
            if ret:
                flow_frame = cv2.cvtColor(three_channeled_flow_frame, cv2.COLOR_BGR2GRAY)
                flow_frames.append(flow_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()

        yield flow_frames


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, video_data, model_type, input_shape, num_classes, batch_size,
                 optical_flow_status, augmentation_status, augmentation_frequency, shuffle):
        'Initialization'
        self.input_shape = input_shape
        self.model_type = model_type
        self.batch_size = batch_size
        self.video_data = video_data
        self.num_classes = num_classes
        self.optical_flow_status = optical_flow_status
        self.augmentation_status = augmentation_status
        self.augmentation_frequency = augmentation_frequency
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation_status == "augmented_onTheFly":
            return int(np.ceil((self.video_data.count()[0] * self.augmentation_frequency) / self.batch_size))
        else:
            return int(np.ceil(self.video_data.count()[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.model_type == "TWOSTREAM_I3D":
            # Find list of IDs
            list_rgbpaths_temp = [os.path.join(self.video_data['rgbclips_path'].values[video_index].strip()) for
                                  video_index in batch_indexes]
            list_xflowpaths_temp = [os.path.join(self.video_data['x_axis_flowclips_path'].values[video_index].strip())
                                    for video_index in batch_indexes]
            list_yflowpaths_temp = [os.path.join(self.video_data['y_axis_flowclips_path'].values[video_index].strip())
                                    for video_index in batch_indexes]
            list_labels_temp = [self.video_data['class'].values[label_index] for label_index in batch_indexes]

            # Generate data
            clipsBatch, labelsBatch = self.__data_generation(list_rgbpaths_temp, list_xflowpaths_temp,
                                                             list_yflowpaths_temp, list_labels_temp)
        else:  # C3D / I3D
            list_rgbpaths_temp = [os.path.join(self.video_data['rgbclips_path'].values[video_index].strip()) for
                                  video_index in batch_indexes]
            list_labels_temp = [self.video_data['class'].values[label_index] for label_index in batch_indexes]

            # Generate data
            clipsBatch, labelsBatch = self.__data_generation(list_rgbpaths_temp, None, None, list_labels_temp)

        return clipsBatch, labelsBatch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.video_data.count()[0])
        if self.augmentation_status == "augmented_onTheFly":
            self.indexes = np.tile(self.indexes, self.augmentation_frequency)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_rgbpaths_temp, list_xflowpaths_temp, list_yflowpaths_temp, list_labels_temp):
        'Generates data containing batch_size samples'

        if self.model_type == "TWOSTREAM_I3D":
            # Initialization
            rgb_channels = 3
            flow_channels = 2
            rgb_clips = np.empty(
                [self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], rgb_channels],
                dtype=np.float32)
            flow_clips = np.empty(
                [self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], flow_channels],
                dtype=np.float32)
            labelsBatch = np.empty((self.batch_size), dtype=int)

            # Generate data
            for numClip, (rgbpath, xflowpath, yflowpath, label) in enumerate(zip(list_rgbpaths_temp,
                                                                                 list_xflowpaths_temp,
                                                                                 list_yflowpaths_temp,
                                                                                 list_labels_temp)):
                # Store sample
                if numClip < self.video_data.count()[0]:
                    rgb_clips[numClip,], flow_clips[numClip,] = get_twostream_videoclip(
                        rgbpath,
                        [xflowpath,
                         yflowpath],
                        self.input_shape[0], self.input_shape[1], self.input_shape[2],
                        self.optical_flow_status, "non_augmented")
                else:
                    rgb_clips[numClip,], flow_clips[numClip,] = get_twostream_videoclip(
                        rgbpath,
                        [xflowpath,
                         yflowpath],
                        self.input_shape[0], self.input_shape[1], self.input_shape[2],
                        self.optical_flow_status, self.augmentation_status)

                # Store class
                labelsBatch[numClip] = label

            clipsBatch = [rgb_clips, flow_clips]

            return clipsBatch, keras.utils.to_categorical(labelsBatch, num_classes=self.num_classes)
        else:  # C3D / I3D
            # Initialization
            rgb_channels = 3
            clipsBatch = np.empty(
                [self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2], rgb_channels],
                dtype=np.float32)
            labelsBatch = np.empty((self.batch_size), dtype=int)

            # Generate data
            for numClip, (rgbpath, label) in enumerate(zip(list_rgbpaths_temp, list_labels_temp)):
                # Store sample
                if numClip < self.video_data.count()[0]:
                    clipsBatch[numClip,] = get_onestream_videoclip(
                        rgbpath,
                        self.input_shape[0], self.input_shape[1], self.input_shape[2],
                        "non_augmented")
                else:
                    clipsBatch[numClip,] = get_onestream_videoclip(
                        rgbpath,
                        self.input_shape[0], self.input_shape[1], self.input_shape[2],
                        self.augmentation_status)

                # Store class
                labelsBatch[numClip] = label

            return clipsBatch, keras.utils.to_categorical(labelsBatch, num_classes=self.num_classes)

##################################################################
# Architecture creation functions
##################################################################

## TwoStream I3D and I3D architecture

def _obtain_input_shape(input_shape,
                        default_frame_size,
                        min_frame_size,
                        default_num_frames,
                        min_num_frames,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_frame_size: default input frames(images) width/height for the model.
        min_frame_size: minimum input frames(images) width/height accepted by the model.
        default_num_frames: default input number of frames(images) for the model.
        min_num_frames: minimum input number of frames accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
            If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
            input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_num_frames, default_frame_size, default_frame_size)
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)
    if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[0] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[1] is not None and input_shape[1] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[2] is not None and input_shape[2] < min_frame_size) or
                        (input_shape[3] is not None and input_shape[3] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                        (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None, None)
            else:
                input_shape = (None, None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias=False,
              use_activation_fn=True,
              use_bn=True,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x


def Inception_Inflated3d(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=None,
                         dropout_prob=0.0,
                         endpoint_logit=True,
                         classes=400,
                         type="rgb"):
    """Instantiates the Inflated 3D Inception v1 architecture.

    Optionally loads weights pre-trained
    on Kinetics. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input frame(image) size for this model is 224x224.

    # Arguments
        include_top: whether to include the the classification
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset only).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)` (with `channels_last` data format)
            or `(NUM_FRAMES, 3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
            NUM_FRAMES should be no smaller than 8. The authors used 64
            frames per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer.
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    x = Inception_architecture(img_input,
                  channel_axis,
                  include_top,
                  dropout_prob,
                  endpoint_logit,
                  classes,
                  type='rgb')

    inputs = img_input
    # create model
    model = Model(input=inputs, output=x, name='i3d_inception')

    # load weights
    if weights in WEIGHTS_NAME:
        if weights == WEIGHTS_NAME[0]:  # rgb_kinetics_only
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[1]:  # flow_kinetics_only
            if include_top:
                weights_url = WEIGHTS_PATH['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[2]:  # rgb_imagenet_and_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'

        elif weights == WEIGHTS_NAME[3]:  # flow_imagenet_and_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'

        model.summary()

        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your keras config '
                          'at ~/.keras/keras.json.')

        x = model.layers[-1].output
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        model = Model(input=inputs, output=x, name='i3d_inception')

    elif weights is not None:
        model.load_weights(weights)
        x = model.layers[-1].output
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        model = Model(input=inputs, output=x, name='i3d_inception')

    else: #No Weights
        x = model.layers[-1].output
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        model = Model(input=inputs, output=x, name='i3d_inception')

    return model

def assign_tuple_value(source_tuple, index, value):
    """
    Assign a value to tuple at the indicated position
    :param tuple: the tuple to alter
    :param index: position of the value to set
    :param value: the value to set
    :return: modified tuple
    """
    temp_list = list(source_tuple)
    temp_list[index] = value
    return tuple(temp_list)

def TwoStream_Inception_Inflated3d(include_top=True,
                         weights=None,
                         input_tensor=None,
                         flow_input_shape=None,
                         rgb_input_shape=None,
                         dropout_prob=0.0,
                         endpoint_logit=True,
                         classes=400):

    """

    :param include_top:
    :param weights: List of two pre_trained model weights RGB and FLOW respectively
    :param input_tensor:
    :param flow_input_shape:
    :param rgb_input_shape:
    :param dropout_prob:
    :param endpoint_logit:
    :param classes:
    :return:
    """

    # Determine flow and rgb input shapes
    flow_input_shape = assign_tuple_value(flow_input_shape, 3, 2)
    flow_input_shape = _obtain_input_shape(
        flow_input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    rgb_input_shape = assign_tuple_value(rgb_input_shape, 3, 3)
    rgb_input_shape = _obtain_input_shape(
        rgb_input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        rgb_img_input = Input(shape=rgb_input_shape)
        flow_img_input = Input(shape=flow_input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            rgb_img_input = Input(tensor=input_tensor, shape=rgb_input_shape)
            flow_img_input = Input(tensor=input_tensor, shape=flow_input_shape)
        else:
            rgb_img_input = input_tensor
            flow_img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4


    flow_x = Inception_architecture(flow_img_input,
                  channel_axis,
                  include_top,
                  dropout_prob,
                  endpoint_logit,
                  classes,
                  type='flow')

    rgb_x = Inception_architecture(rgb_img_input,
                           channel_axis,
                           include_top,
                           dropout_prob,
                           endpoint_logit,
                           classes,
                          type='rgb')
    # create model
    FLOW_stream = Model(input=flow_img_input, output=flow_x, name='flow_stream_i3d_inception')
    RGB_stream = Model(input=rgb_img_input, output=rgb_x, name='rgb_stream_i3d_inception')

    # load weights
    if weights in WEIGHTS_NAME:
        if weights == WEIGHTS_NAME[0, 1]:  # rgb_kinetics_only and flow_kinetics_only
            if include_top:
                rgb_weights_url = WEIGHTS_PATH['rgb_kinetics_only']
                rgb_model_name = 'i3d_inception_rgb_kinetics_only.h5'
                flow_weights_url = WEIGHTS_PATH['flow_kinetics_only']
                flow_model_name = 'i3d_inception_flow_kinetics_only.h5'
            else:
                rgb_weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
                rgb_model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'
                flow_weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
                flow_model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[2, 3]:  # rgb_imagenet_and_kinetics and flow_imagenet_and_kinetics
            if include_top:
                rgb_weights_url = WEIGHTS_PATH['rgb_imagenet_and_kinetics']
                rgb_model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
                flow_weights_url = WEIGHTS_PATH['flow_imagenet_and_kinetics']
                flow_model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
            else:
                rgb_weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
                rgb_model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
                flow_weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
                flow_model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'

        downloaded_rgb_weights_path = get_file(rgb_model_name, rgb_weights_url, cache_subdir='models')
        downloaded_flow_weights_path = get_file(flow_model_name, flow_weights_url, cache_subdir='models')
        RGB_stream.load_weights(downloaded_rgb_weights_path)
        FLOW_stream.load_weights(downloaded_flow_weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(RGB_stream)
            layer_utils.convert_all_kernels_in_model(FLOW_stream)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your keras config '
                          'at ~/.keras/keras.json.')

        x = RGB_stream.layers[-1].output
        x = Flatten()(x)

        y = FLOW_stream.layers[-1].output
        y = Flatten()(y)

    elif weights is not None:
        RGB_stream.load_weights(weights[0])
        FLOW_stream.load_weights(weights[1])

        x = RGB_stream.layers[-1].output
        x = Flatten()(x)

        y = FLOW_stream.layers[-1].output
        y = Flatten()(y)

    else: #No Weights
        x = RGB_stream.layers[-1].output
        x = Flatten()(x)

        y = FLOW_stream.layers[-1].output
        y = Flatten()(y)

    global_stream = layers.concatenate([x, y])
    global_stream = Dense(classes, activation='softmax', name='predictions')(global_stream)

    model = Model(input=[rgb_img_input, flow_img_input], output=global_stream, name='TwoStream_I3D')

    return model

def Inception_architecture(img_input,
                  channel_axis,
                  include_top,
                  dropout_prob,
                  endpoint_logit,
                  classes,
                  type):

    if type == 'flow':
        layer_name_extension = '_flow'
    else:
        layer_name_extension = '_rgb'
    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7'+layer_name_extension)

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3'+layer_name_extension)(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1'+layer_name_extension)
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3'+layer_name_extension)

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3'+layer_name_extension)(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b'+layer_name_extension)

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c'+layer_name_extension)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3'+layer_name_extension)(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b'+layer_name_extension)

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c'+layer_name_extension)

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d'+layer_name_extension)

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e'+layer_name_extension)

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f'+layer_name_extension)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2'+layer_name_extension)(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b'+layer_name_extension)

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1'+layer_name_extension)

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1'+layer_name_extension)
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3'+layer_name_extension)

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1'+layer_name_extension)
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3'+layer_name_extension)

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3'+layer_name_extension)(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1'+layer_name_extension)

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c'+layer_name_extension)


    if include_top:
        # Classification block
        # We maybe consider to remove this useless code snippet and keep only what is considered in the else statement
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool'+layer_name_extension)(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                      use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1'+layer_name_extension)

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction'+layer_name_extension)(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool'+layer_name_extension)(x)

    return x


## C3D architecture

def ConvNets3D(input_shape, summary=False, num_classes=487, backend='tf'):
    """ 
    Return the Keras model of the network
    """
    model = Sequential()

    model.add(Conv3D(64, 3, 3, 3, activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu',
                            padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                            padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                            padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                            padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                            padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                            padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                            padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(num_classes, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())

    return model

## Resnet 3D architecture
# Taken from https://github.com/JihongJu/keras-resnet3d/blob/master/resnet3d/resnet3d.py

def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    # kernel_initializer = conv_params.setdefault(
    #     "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, 
                      # kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      padding=padding)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    # kernel_initializer = conv_params.setdefault("kernel_initializer",
    #                                             "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, 
                      # kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      padding=padding)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = math.ceil(int(input.shape[DIM1_AXIS]) \
        / int(residual.shape[DIM1_AXIS]))
    stride_dim2 = math.ceil(int(input.shape[DIM2_AXIS]) \
        / int(residual.shape[DIM2_AXIS]))
    stride_dim3 = math.ceil(int(input.shape[DIM3_AXIS]) \
        / int(residual.shape[DIM3_AXIS]))
    equal_channels = int(residual.shape[CHANNEL_AXIS]) \
        == int(input.shape[CHANNEL_AXIS])

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=int(residual.shape[CHANNEL_AXIS]),
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            # kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            padding="valid"
            )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, 
                      kernel_regularizer,
                      repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides,
                           # kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer,
                           padding="same"
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    kernel_regularizer=kernel_regularizer,
                                    strides=strides
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, 
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_size=(3, 3, 3)
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, 
                              # kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer,
                              padding="same"
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       kernel_regularizer=kernel_regularizer,
                                       strides=strides
                                       )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_size=(3, 3, 3)
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_size=(1, 1, 1)
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        print("here CHANNELS last")
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                kernel_regularizer=l2(reg_factor),
                                strides=(2, 2, 2)
                                )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                             padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(int(block.shape[DIM1_AXIS]),
                                            int(block.shape[DIM2_AXIS]),
                                            int(block.shape[DIM3_AXIS])),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          # kernel_initializer="he_normal",
                          kernel_regularizer=l2(reg_factor),
                          activation="softmax"
                          
                          )(flatten1)
        else:
            dense = Dense(units=num_outputs,
                          # kernel_initializer="he_normal",
                          kernel_regularizer=l2(reg_factor),
                          activation="sigmoid"
                          )(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        regularization_factor = 1e-4
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2], reg_factor=regularization_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        regularization_factor = 1e-4
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], reg_factor=regularization_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 50."""
        regularization_factor = 1e-4
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 6, 3], reg_factor=regularization_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 101."""
        regularization_factor = 1e-4
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 23, 3], reg_factor=regularization_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 152."""
        regularization_factor = 1e-4
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 8, 36, 3], reg_factor=regularization_factor)


##################################################################
# Training and evaluation utils functions
##################################################################

def define_input(model_type):
    """
    Defines the input prototype according to the chosen model type to train
    :param model_type: type of the model to evaluate
    :return: returns the input prototype
    """

    if model_type == 'I3D':
        print("# I3D sample_input creation :")
        FRAMES_PER_VIDEO = 20
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        FRAME_CHANNEL = 3

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    elif model_type == 'TWOSTREAM_I3D':
        print("# TWOSTREAM_I3D sample_input creation :")
        FRAMES_PER_VIDEO = 20
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        FRAME_CHANNEL = 0

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    elif model_type == 'C3D':
        print("# C3D sample_input creation :")
        FRAMES_PER_VIDEO = 16
        FRAME_HEIGHT = 112
        FRAME_WIDTH = 112
        FRAME_CHANNEL = 3

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    elif model_type == 'R3D_18' or model_type == 'R3D_34' or model_type == 'R3D_50' or model_type == 'R3D_101' or model_type == 'R3D_152':
        print("# R3D sample_input creation :")
        FRAMES_PER_VIDEO = 16
        FRAME_HEIGHT = 112
        FRAME_WIDTH = 112
        FRAME_CHANNEL = 3

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    else:
        print("Unknown model")

    return sample_input


def train_load_model(model_type, training_condition, input_shape, nb_classes):
    """
    Prepares the model for training
    :param model_type: Name of the architecture
    :param training_condition: If to load weights (pretrained) or no (scratch)
    :param input_shape: structure of the input shape
    :param nb_classes: number of classes
    :return: return the defined model
    """

    if model_type == 'I3D':
        print("I3D training")
        if training_condition == '_PRETRAINED':
            model_weights_rgb = "Trained_models/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5"
            model = Inception_Inflated3d(include_top=False,
                                         weights=model_weights_rgb,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        else: #_SCRATCH
            model = Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
    elif model_type == 'TWOSTREAM_I3D':
        print("TwoStream I3D training")
        if training_condition == '_PRETRAINED':
            model_weights_rgb = "Trained_models/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5"
            model_weights_flow =  "Trained_models/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5"
            model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=[model_weights_rgb, model_weights_flow],
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        else: #_SCRATCH
            model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
    elif model_type == 'C3D':
        print("C3D training")
        if training_condition == '_PRETRAINED':
            model_weights_rgb = "Trained_models/sports1M_weights_tf.h5"
            model = ConvNets3D(input_shape, num_classes=487, backend='tf')
            # Prepare for fine-tuning
            model.load_weights(model_weights_rgb)
            model.pop()
            model.add(Dense(nb_classes, activation='softmax', name='predictions'))
        else : #_SCRATCH
            model = ConvNets3D(input_shape,
                              num_classes=nb_classes,
                              backend='tf')
    elif model_type == 'R3D_18':
        print("R3D with 18 layers training")
        print("Training from scratch")
        #_SCRATCH
        model = Resnet3DBuilder.build_resnet_18(input_shape, nb_classes)
    elif model_type == 'R3D_34':
        print("R3D with 34 layers training")
        print("Training from scratch")
        #_SCRATCH
        model = Resnet3DBuilder.build_resnet_34(input_shape, nb_classes)
    elif model_type == 'R3D_50':
        print("R3D with 50 layers training")
        print("Training from scratch")
        #_SCRATCH
        model = Resnet3DBuilder.build_resnet_50(input_shape, nb_classes)
    elif model_type == 'R3D_101':
        print("R3D with 101 layers training")
        print("Training from scratch")
        #_SCRATCH
        model = Resnet3DBuilder.build_resnet_101(input_shape, nb_classes)
    elif model_type == 'R3D_152':
        print("R3D with 152 layers training")
        print("Training from scratch")
        #_SCRATCH
        model = Resnet3DBuilder.build_resnet_152(input_shape, nb_classes)
    else:
        print("Unknown model")
    return model

def evaluate_load_model(model_type, model_weights_path, input_shape, nb_classes):
    """
    Loads a model from the models folder
    :param model_type: model type to load
    :param model_weights_path: path to the trained model
    :param input_shape: prototype of the input data
    :param nb_classes: the number of classes of the dataset
    :return: returns the loaded model
    """

    if model_type == 'I3D':
        print("I3D evaluation")
        model = Inception_Inflated3d(include_top=False,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=input_shape,
                                     dropout_prob=0.0,
                                     endpoint_logit=True,
                                     classes=nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'TWOSTREAM_I3D':
        print("TWOSTREAM_I3D evaluation")
        model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'C3D':
        print("C3D evaluation")
        model = ConvNets3D(input_shape,
                          num_classes=nb_classes,
                          backend='tf')

        model.load_weights(model_weights_path)
    elif model_type == 'R3D_18':
        print("R3D with 18 layers evaluation")
        model = Resnet3DBuilder.build_resnet_18(input_shape,nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'R3D_34':
        print("R3D with 34 layers evaluation")
        model = Resnet3DBuilder.build_resnet_34(input_shape, nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'R3D_50':
        print("R3D with 50 layers evaluation")
        model = Resnet3DBuilder.build_resnet_50(input_shape, nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'R3D_101':
        print("R3D with 101 layers evaluation")
        model = Resnet3DBuilder.build_resnet_101(input_shape, nb_classes)
        model.load_weights(model_weights_path)
    elif model_type == 'R3D_152':
        print("R3D with 152 layers evaluation")
        model = Resnet3DBuilder.build_resnet_152(input_shape, nb_classes)
        model.load_weights(model_weights_path)
    else:
        print("Unknown model")
    return model

def scheduler(epoch, learningrate):
    """
    Scheduler used to reduce the learning rate each 4 epochs
    :param epoch: current epoch index
    :param lr: current learning rate
    :return: new learning rate
    """
    if epoch % 4 == 0 and epoch != 0:
        learningrate = learningrate/10
    return learningrate


def train(model_type, 
        training_condition,
        train_path, 
        val_path, 
        batch_size, 
        epochs, 
        workers, 
        optical_flow_status, 
        augmentation_status,
        augmentation_frequency,
        classes_status, 
        model_path):
    """
    Trains a model from scratch or fine-tunes it and then evaluate it
    :param model_type: Name of the architecture to train
    :param training_condition: Type of the training : from scratch or fine-tuning
    :param data_folder: Folder containing the folder
    :param batch_size: batch size
    :param epochs: the number of epochs
    :param optical_flow_status : says if the optical flow computation was pre-computed (TVL1) or done on-the-fly (FarneBack)
    :param augmentation_status : says if the data is augmented or not, on-the-fly or pre_computed
    :param augmentation_frequency : number that determines to which extent the data was pre-augmented
    :param classes_status : says if the classes will be balanced during the training phase or not
    :return: a trained model
    """
    # Specify sample_input
    sample_input = define_input(model_type)

    # Read Dataset
    if augmentation_status != "augmented_precomputed":
        train_data = pd.read_csv(train_path)
    else:
        # In the dataframe add augmented_i next to rgb_videoclips and their TV_L1 conversion next to flow_videoclips
        # Loop over the frequency information
        train_dataframe = pd.read_csv(train_path)
        train_data = augment_dataframe(train_dataframe, augmentation_frequency)

    validation_data = pd.read_csv(val_path)

    nb_classes = len(set(train_data['class']))

    video_train_generator = DataGenerator(train_data,
                                         model_type,
                                         sample_input.shape,
                                         nb_classes,
                                         batch_size,
                                         optical_flow_status,
                                         augmentation_status,
                                         augmentation_frequency,
                                         shuffle=True)
    video_val_generator = DataGenerator(validation_data,
                                       model_type,
                                       sample_input.shape,
                                       nb_classes,
                                       batch_size,
                                       optical_flow_status,
                                       augmentation_status="non_augmented",
                                       augmentation_frequency=0,
                                       shuffle=True)

    # Get Model
    model = train_load_model(model_type, training_condition, sample_input.shape, nb_classes)

    # Callbacks
    checkpoint = ModelCheckpoint(model_path+'_weights.hdf5',
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=100)

    if model_type == 'C3D':
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=200,
                                           verbose=1,  mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
        change_lr = LearningRateScheduler(scheduler, verbose=1)
        callbacks_list = [change_lr, checkpoint, reduceLROnPlat, earlyStop]
    elif model_type == 'R3D':
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                           patience=1, verbose=1, mode='min', min_lr=1e-4)
        callbacks_list = [checkpoint, reduceLROnPlat, earlyStop]
    else: # I3D and TWOSTREAM_I3D
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=0, mode='min',
                                           verbose=1,min_lr=1e-6)
        callbacks_list = [checkpoint, reduceLROnPlat, earlyStop]


    # compile model
    if model_type == 'I3D' or model_type == 'TWOSTREAM_I3D':
        optim = SGD(lr=0.003, momentum=0.9)
    elif model_type == 'C3D':
        optim = SGD(lr=0.003)
    elif model_type == 'R3D_18' or model_type == 'R3D_34' or model_type == 'R3D_50' or model_type == 'R3D_101' or model_type == 'R3D_152':
        # optim = Adam(lr=0.1)
        optim = Adam(lr=1e-3)
    else:
        optim = SGD(lr=0.003, momentum=0.9, nesterov=True, decay=1e-6)

    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(model_path+'_weights.hdf5'):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights(model_path+'_weights.hdf5')
        print('Weights loaded')

    # model description
    model.summary()

    # train model
    print('Training started....')

    use_multiprocessing = False

    if classes_status == "balanced":
        class_weights = class_weight.compute_class_weight(classes_status,
                                                 np.unique(train_data['class']),
                                                 train_data['class'])
        history = model.fit_generator(
            generator=video_train_generator,
            validation_data=video_val_generator,
            verbose=1,
            callbacks=callbacks_list,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            epochs=epochs,
            class_weight=class_weights)
    else:
        history = model.fit_generator(
            generator=video_train_generator,
            validation_data=video_val_generator,
            verbose=1,
            callbacks=callbacks_list,
            workers=workers,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing)

    return history

def evaluate(model_type, trained_model_name, test_path, batch_size, workers, optical_flow_status, augmentation_status):
    """
    Evaluates a trained model on the test set of Crowd-11
    :param model_type: the model type to train, it must be stored on a .hdf5 file
    :param trained_model_name: mentions the path to the trained model
    :param test_data: load the test set
    :return: prints the accuracy and the loss ['loss', 'acc']
    """
    # Load data
    test_data = pd.read_csv(test_path)

    # Determine the number of classes
    nb_classes = len(set(test_data['class']))

    # Define input
    sample_input = define_input(model_type)

    # Load trained model
    trained_model_path = trained_model_name + "_weights.hdf5"
    model = evaluate_load_model(model_type, trained_model_path, sample_input.shape, nb_classes)
    model.summary()
    if model_type == 'R3D_18' or model_type == 'R3D_34' or model_type == 'R3D_50' or model_type == 'R3D_101' or model_type == 'R3D_152':
        # model.compile(optimizer=Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
    else:
        model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy',metrics=['accuracy'])

    # Evaluate the model
    video_test_generator = DataGenerator(test_data,
                           model_type,
                           sample_input.shape,
                           nb_classes,
                           batch_size=1,
                           optical_flow_status=optical_flow_status,
                           augmentation_status="non_augmented",
                           augmentation_frequency=0,
                           shuffle=False)

    use_multiprocessing = False

    [test_loss, test_acc] = model.evaluate_generator(
                            generator=video_test_generator,
                            workers=1,
                            use_multiprocessing=use_multiprocessing,
                            verbose=1)

    return test_loss, test_acc


##################################################################
# Main program
##################################################################

def main(args):
    try:
        print(args.train_path)
        print(args.val_path)
        print(args.test_path)
        split_specification = "_" + os.path.basename(os.path.dirname(args.train_path))
        TestSplit_subfolder = "TestSplit" + re.findall(r'%s(\d+)' % "test", split_specification)[0]
        trained_models_subfolder_name =  str(args.folds_number) + \
            "folds_" + \
            args.model_type + \
            args.training_condition + \
            "_CS_" + \
            args.classes_status + \
            "_OF_" + \
            args.optical_flow_status + \
            "_AS_" + \
            args.augmentation_status

        if args.augmentation_status == "augmented_precomputed":
            trained_model_name = os.path.join(args.trained_models_folder,
                trained_models_subfolder_name,
                TestSplit_subfolder,
                trained_models_subfolder_name + \
                "_Freq"+str(args.augmentation_frequency) + \
                split_specification)
        else:
            trained_model_name = os.path.join(args.trained_models_folder,
                trained_models_subfolder_name,
                TestSplit_subfolder,
                trained_models_subfolder_name + \
                split_specification)

        # Train a model
        history_ = train(args.model_type,
              args.training_condition,
              args.train_path,
              args.val_path,
              args.batch_size,
              args.epochs,
              args.workers,
              args.optical_flow_status,
              args.augmentation_status,
              args.augmentation_frequency,
              args.classes_status,
              trained_model_name)

        # Evaluate the trained model
        test_loss, test_acc = evaluate(args.model_type,
                trained_model_name,
                args.test_path,
                args.batch_size,
                args.workers,
                args.optical_flow_status,
                args.augmentation_status)
        
        print("Val_acc : ", history_.history['val_acc'])
        print("Val_loss : ", history_.history['val_loss'])
        print("Test_acc : ", test_acc)
        print("Test_loss : ", test_loss)

        # Store the history containing the validation loss and accuracy of the best model
        store_history(args.ensemble_models_weights_folder,
                args.folds_number,
                split_specification,
                trained_model_name,
                history_)

    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()


if __name__ == '__main__':
    ## ensure that the script is running on gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    ## clear session, in case it's necessary
    K.clear_session()

    ## verify that we are running on gpu
    if len(K.tensorflow_backend._get_available_gpus()) == 0:
        print('error-no-gpu')
        exit()
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn',
                        '--folds_number',
                        help='Specify the number of folds.',
                        type=int,
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

    parser.add_argument('-cs',
                        '--classes_status',
                        help='Mentions if we want to make the data balanced or keep it as is.',
                        choices=['balanced', 'unbalanced'],
                        type=str,
                        default='balanced',
                        required=True)

    parser.add_argument('-trp',
                        '--train_path',
                        help='Specify the path to the train csv file.',
                        type=str,
                        required=True)

    parser.add_argument('-tsp',
                        '--test_path',
                        help='Specify the path to the test csv file.',
                        type=str,
                        required=True)

    parser.add_argument('-vp',
                        '--val_path',
                        help='Specify the path to the validation csv file.',
                        type=str,
                        required=True)

    parser.add_argument('-tmf',
                        '--trained_models_folder',
                        help='Specify the path to the trained models.',
                        type=str,
                        default='Trained_models/',
                        required=True)

    parser.add_argument('-b',
                        '--batch_size',
                        help='Specify the batch_size for training.',
                        type=int,
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
                        help='Associated with augmentation_status when it is set to augmented_precomputed.',
                        type=int,
                        default=0)

    parser.add_argument('-ofs',
                        '--optical_flow_status',
                        help='Specifies if the optical flow was pre-computed (happens with TV-L1) or is computed on-the-fly (happens with Farneback).',
                        type=str,
                        choices=['TVL1_precomputed', 'FarneBack_onTheFly'],
                        required=True)

    parser.add_argument('-emwf',
                        '--ensemble_models_weights_folder',
                        help='Path to the weights folder.',
                        type=str,
                        default='Data/Weights',
                        required=True)

    parser.add_argument('-e',
                        '--epochs',
                        help='Specify the number of epochs for training.',
                        type=int,
                        required=True)

    parser.add_argument('-w',
                        '--workers',
                        help='Specify the number of GPUs involved in the computation.',
                        type=int,
                        required=True)
    
    args = parser.parse_args()

    main(args)