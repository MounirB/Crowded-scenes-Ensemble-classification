# Crowded scenes Ensemble classification

This is the companion repository for our paper titled "Classification ensembliste de vidéos de mouvements de foule" accepted for presentation at the french national conference [ORASIS 2021](https://orasis2021.sciencesconf.org/) and our paper titled "Ensemble classification of video-recorded crowd movements" accepted for presentation in the [ISPA 2021](https://www.isispa.org/) IEEE conference.
The project is about the Ensemble classification of 10 crowd movements illustrated in the Crowd-11 dataset. The 11th class is intended for empty scenes.

Four different architectures are employed for the Ensemble classification: 
- The C3D architecture, namely the 3D ConvNets that is presented in the following article: [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf).
```
@inproceedings{tran2015learning,
  title={Learning spatiotemporal features with 3d convolutional networks},
  author={Tran, Du and Bourdev, Lubomir and Fergus, Rob and Torresani, Lorenzo and Paluri, Manohar},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4489--4497},
  year={2015}
}
```
The implementation of C3D in Keras was forked from [here](https://github.com/axon-research/c3d-keras).

- The I3D architecture and its extension the TwoStream-I3D. Namely the Inflated 3D architecture that is presented in the following article: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

```
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
```

The implementation of I3D in Keras was forked from [here](https://github.com/dlpbc/keras-kinetics-i3d). According to the authors of the [keras-kinetics-i3d](https://github.com/dlpbc/keras-kinetics-i3d) repository, the weights of I3D, that we provide in the `Data/` folder, were obtained from [here](https://github.com/dlpbc/keras-kinetics-i3d) and are under [Apache-2.0 License](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE).

- The R3D architecture, namely the ResNet 3D architecture that is presented in the following article: [Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Hara_Learning_Spatio-Temporal_Features_ICCV_2017_paper.pdf).

```
@inproceedings{hara2017learning,
  title={Learning spatio-temporal features with 3d residual networks for action recognition},
  author={Hara, Kensho and Kataoka, Hirokatsu and Satoh, Yutaka},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={3154--3160},
  year={2017}
}
```

We used the keras implementation of the R3D architecture that can be found [here](https://github.com/JihongJu/keras-resnet3d).

# Requirements

Refer to the `requirements.txt` file to install the required versions of tensorflow-gpu, Keras, Opencv, Numpy. You may also need to install Matplotlib, Pandas, Scikit-learn, Scikit-videos, Scikit-image.

## Download the Crowd-11 dataset

Instructions on how to get the Crowd-11 dataset may be found in the following workshop paper : [Crowd-11: A Dataset for Fine Grained Crowd Behaviour Analysis](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w37/papers/Dupont_Crowd-11_A_Dataset_CVPR_2017_paper.pdf)

```
@inproceedings{dupont2017crowd,
  title={Crowd-11: A dataset for fine grained crowd behaviour analysis},
  author={Dupont, Camille and Tobias, Luis and Luvison, Bertrand},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={9--16},
  year={2017}
}
```

# Usage

After downloading the Crowd-11 dataset, extract the optical flow and put both of the flow and the rgb clips inside the subfolders of the `Data/Crowd-11/` folder, like this:

```
-- Data/
    -- Crowd-11/
        -- rgb/
        -- flow/
```

The optical flow is extracted using the following forked and updated project: https://github.com/MounirB/py-denseflow

When you obtain the Crowd-11 dataset, it is crucial to include the `preprocessing.csv` file into the `Data` folder before generating the k Folds.

## Generate k Folds
Before creating homogeneous Ensembles based on the dataset folds, you should split the dataset into several folds. To do so, launch the following script `generate_folds.sh` that will run the `generate_folds.py` script. By default, the number of folds K is set to 5.
In the main program of the `generate_folds.py` script, you can find that the program relies on `preprocessing.csv` spreadsheet. 
Remove, if needed, the missing clips paths of Crowd-11 form `preprocessing.csv`.

## Data augmentation

To augment the dataset folds, use the `augment_dataset.sh` script. In this script, you should specify the augmentation frequency `--augmentation_frequency`.
The augmentation script `augment_dataset.py` is forked from this project: https://github.com/okankop/vidaug

## Train Ensembles
To train an Ensemble of homogeneous models, run the following script `launch_train_ensemble.sh`.
The options in the script will guide you to choose the unique architecture of the models of the Ensemble.

## Evaluate Ensembles of Homogeneous models
To evaluate an Ensemble of homogeneous models, run the following script `launch_evaluate_ensemble.sh`.
The options in the script will guide you to choose the pre-trained Ensemble to evaluate.

## Evaluate Global Ensembles of heterogeneous Ensembles of homogeneous models
To evaluate a Global Ensemble of heterogeneous Ensembles of homogeneous models or evaluate different combinations, run once again the following script `launch_evaluate_ensemble.sh`.
