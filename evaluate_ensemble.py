# coding=utf-8

from train import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
from itertools import product
import matplotlib.pyplot as plt
import glob
import ast
import re
import itertools

#########################################################################################
## Utils functions
#########################################################################################
def getModelTypeAndTrainingCondition(model_name):
    """
    Extract the model_type and the training_condition from the model_name
    :param model_name: complete string or path containing the model type and the training condition
    :return: model_type, training_condition
    """
    # Get the training condition of the model
    training_condition_regex = "(_PRETRAINED|_SCRATCH)"
    training_condition = re.search(training_condition_regex, model_name)[0]

    # Get the model type
    model_type_regex = "(TWOSTREAM_I3D|I3D|C3D|R3D_18|R3D_34|R3D_50|R3D_101|R3D_152)"
    model_type = re.search(model_type_regex, model_name)[0]

    return model_type, training_condition

def get_modeltraining_validation_loss(histories_folder, test_index):
    """
    Return the model weights according the results obtained in the training history
    Compute the validation error inverse
    """
    # Determine the number of folds from the histories_folder filename
    nb_folds = int(os.path.basename(histories_folder)[0])
    # Determine the validation indexes using the nb_folds and excluding the test_index
    val_folds_indices = list(range(0, nb_folds))
    val_folds_indices.remove(test_index)
    # Declare the weights array
    weights = list()
    # Determine the testSplit_subfolder filename
    testSplit_subfolder = "TestSplit" + str(test_index)
    history_subfolder = os.path.join(histories_folder, testSplit_subfolder)
    histories_list = os.listdir(history_subfolder)
    history_filename = None
    # Retrieve the validation error 
    for val_index in val_folds_indices:
        split_specification = "split_test" + str(test_index) + "_val" + str(val_index)
        history_filename = \
        [history_file for history_file in histories_list if re.search(split_specification, history_file)][0]
        val_losses = np.load(os.path.join(history_subfolder, history_filename))
        min_val_loss = np.min(val_losses)
        weights.append(1 / min_val_loss)

    # Compute the validation error inverse
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    return np.array(weights)


def convert_str2array(raw_probabilities_str):
    """
    Converts probabilities values stored in string format to float
    """
    raw_probabilities_str_cleaned = raw_probabilities_str.replace("array(", "")
    raw_probabilities_str_cleaned = raw_probabilities_str_cleaned.replace(", dtype=float32)", "")
    probabilities_list = ast.literal_eval(raw_probabilities_str_cleaned)
    probabilities = np.array(probabilities_list)
    return probabilities


def convert_array2listofarrays(probabilities_array):
    """
    Converts probabilities values from an array of arrays to a list of arrays
    """
    list_probabilities = list()
    for probabilities in probabilities_array:
        list_probabilities.append(probabilities)
    return list_probabilities


def evaluate_single_model(trained_model_path, test_labels, probabilities_file, nb_classes):
    """
    Returns the predictions of a single model based on the precomputed probabilities
    """
    # make predictions
    probabilities = pd.read_csv(probabilities_file)
    # Get the probabilities of each model using pandas
    yhat = convert_str2array(
        probabilities.loc[probabilities['path'] == os.path.splitext(trained_model_path)[0]]['probabilities'].values[0])
    yhat = np.reshape(yhat, (len(test_labels), nb_classes))
    # Make predictions based on the maximum probability
    predictions = np.argmax(yhat, axis=1)
    # Evaluate score
    score = accuracy_score(test_labels, predictions)
    return score, predictions


## Global evaluation Utils functions

def get_ModelsNameAndTrainedModelsSubfolder(folds_number,
                                            trained_models_folder,
                                            model_type,
                                            training_condition,
                                            classes_status,
                                            optical_flow_status,
                                            augmentation_status,
                                            augmentation_frequency):
    """
    Return the models_name and the trained_models_subfolder
    :param folds_number: number of folds
    :param trained_models_folder : parent folder of the trained models
    :param model_type: architecture type of the model
    :param training_condition: training condition of the model
    :param classes_status: whether balanced or unbalanced
    :param optical_flow_status: whether it is TVL1_precomputed or FarneBack_onTheFly
    :param augmentation_status: whether it is augmented or non_augmented
    :param augmentation_frequency: (optional) if the data is augmented
    """

    # Create the models name
    if augmentation_status == "augmented_precomputed":
        models_name = str(folds_number) + \
                      "folds_" + \
                      model_type + \
                      training_condition + \
                      "_CS_" + \
                      classes_status + \
                      "_OF_" + \
                      optical_flow_status + \
                      "_AS_" + \
                      augmentation_status + \
                      "_Freq" + str(augmentation_frequency)
    else:
        models_name = str(folds_number) + \
                      "folds_" + \
                      model_type + \
                      training_condition + \
                      "_CS_" + \
                      classes_status + \
                      "_OF_" + \
                      optical_flow_status + \
                      "_AS_" + \
                      augmentation_status
    # Create the trained models subfolder
    trained_models_subfolder = os.path.join(trained_models_folder, models_name)

    return models_name, trained_models_subfolder


def createModelsTrainingConditionsDictionary(models_list):
    """
    creates models_trainingconditions dictionary
    :param models_list: list of models with their training conditions
    :return: dictionary which keys are model types and values training conditions
    """
    model_trainingConditions = dict()
    """
    SPECIALCASE refers to the unique model that benefitted from data augmentation : TWOSTREAM_I3D
    This unique model had FarneBack computed on the fly and its training data was augmented thrice.
    """
    model_types = ['SPECIALCASE', 'TWOSTREAM_I3D', 'C3D', 'I3D', 'R3D_18', 'R3D_34', 'R3D_50', 'R3D_101', 'R3D_152']
    training_conditions = ['_PRETRAINED', '_SCRATCH']
    for model in models_list:
        for model_type in model_types:
            for training_condition in training_conditions:
                if model_type + training_condition == model:
                    if model_type in model_trainingConditions:
                        model_trainingConditions[model_type].append(training_condition)
                    else:
                        model_trainingConditions[model_type] = [training_condition]

    return model_trainingConditions


def lookFor_probabilitiesFile(nb_folds,
                              results_folder,
                              model_type,
                              training_condition,
                              classes_status,
                              optical_flow_status,
                              augmentation_status,
                              augmentation_frequency,
                              involved_sets):
    """
    Looks for the test_probabilities_file
    :param nb_folds: the number of folds
    :param results_folder: results_folder
    :param model_type: the model's architecutre
    :param training_condition: the training condition of the model
    :param classes_status: ['balanced','unbalanced'] for the dataset
    :param optical_flow_status: ['TVL1_precomputed', 'FarneBack_onTheFly']
    :param augmentation_status: ['non_augmented', 'augmented']
    :param augmentation_frequency: augmentation frequency of the dataset if the dataset is augmented
    :param involved_sets: whether "test" set or "train_val" sets
    :return: test_probabilities_file path
    """

    if augmentation_status == "augmented_precomputed":
        test_probabilities_file_path = os.path.join(results_folder,
                                                    involved_sets + "_predicted_probabilities_" + str(
                                                        nb_folds) + 'folds_' + model_type + training_condition + "_CS_" + classes_status + "_OF_" + optical_flow_status + "_AS_" + augmentation_status + "_Freq" + str(
                                                        augmentation_frequency) + ".csv")
    else:
        test_probabilities_file_path = os.path.join(results_folder,
                                                    involved_sets + "_predicted_probabilities_" + str(
                                                        nb_folds) + 'folds_' + model_type + training_condition + "_CS_" + classes_status + "_OF_" + optical_flow_status + "_AS_" + augmentation_status + ".csv")

    if os.path.isfile(test_probabilities_file_path):
        return test_probabilities_file_path
    else:
        return None

def lookFor_UniqueEnsemble_predictionsFile(nb_folds,
                                           results_folder,
                                           model_type,
                                           training_condition,
                                           classes_status,
                                           optical_flow_status,
                                           augmentation_status,
                                           augmentation_frequency):
    """
    Looks for the test_predictions_file for unique ensembles
    :param nb_folds: the number of folds
    :param results_folder: results_folder
    :param model_type: the model's architecutre
    :param training_condition: the training condition of the model
    :param classes_status: ['balanced','unbalanced'] for the dataset
    :param optical_flow_status: ['TVL1_precomputed', 'FarneBack_onTheFly']
    :param augmentation_status: ['non_augmented', 'augmented']
    :param augmentation_frequency: augmentation frequency of the dataset if the dataset is augmented
    :return: test_predictions_file path
    """

    if augmentation_status == "augmented_precomputed":
        test_predictions_file_path = os.path.join(results_folder,
                                                    "weighted_prediction_results_" + str(
                                                        nb_folds) + 'folds_' + model_type + training_condition + "_CS_" + classes_status + "_OF_" + optical_flow_status + "_AS_" + augmentation_status + "_Freq" + str(
                                                        augmentation_frequency) + ".csv")
    else:
        test_predictions_file_path = os.path.join(results_folder,
                                                    "weighted_prediction_results_" + str(
                                                        nb_folds) + 'folds_' + model_type + training_condition + "_CS_" + classes_status + "_OF_" + optical_flow_status + "_AS_" + augmentation_status + ".csv")

    if os.path.isfile(test_predictions_file_path):
        return test_predictions_file_path
    else:
        return None

def lookFor_GlobalEnsemble_predictionsFile(nb_folds,
                                           results_folder,
                                           models_list):
    """
    Looks for the test_predictions_file for global ensembles
    :param nb_folds: the number of folds
    :param results_folder: results_folder
    :param models_list: list of the models involved in the global ensemble
    :return: test_predictions_file path
    """
    separator = '_'
    all_models_names_string = separator.join(models_list)

    test_predictions_file_path = os.path.join(results_folder,
                                              "global_ensemble_summed_prediction_results_" +
                                              str(nb_folds) + "_folds_"
                                              + all_models_names_string + "_.csv")

    if os.path.isfile(test_predictions_file_path):
        return test_predictions_file_path
    else:
        return None

#########################################################################################
## Chunks of code taken from machinelearningmastery used for weights retrieval
#########################################################################################

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = np.linalg.norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# loss function for optimization process, designed to be minimized
def loss_function(weights, members, probabilities_file, testy):
    # normalize weights
    normalized = normalize(weights)
    # compute the number of classes
    nb_classes = len(np.unique(testy))
    # calculate error rate
    return 1.0 - evaluate_ensemble(members, normalized, probabilities_file, testy, nb_classes)[0]


def apply_differential_evolution(n_members, members_paths, probabilities_file, testy):
    # define bounds on each weight
    bound_w = [(0.0, 1.0) for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (members_paths, probabilities_file, testy)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=20, tol=1e-7, disp=True)
    # get the chosen weights
    weights = normalize(result['x'])
    return weights


# grid search weights
def apply_grid_search(members_paths, probabilities_file, testy):
    # get the chosen weights
    weights = grid_search(members_paths, probabilities_file, testy)
    print('Weights: %s' % weights)
    return np.array(weights)


def grid_search(members, testX, testy):
    # define weights to consider
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    nb_classes = len(np.unique(testy))
    # iterate all possible combinations (cartesian product)
    for weights in product(w, repeat=len(members)):
        # skip if all weights are equal
        if len(set(weights)) == 1:
            continue
        # hack, normalize weight vector
        weights = normalize(weights)
        # evaluate weights
        score = evaluate_ensemble(members, weights, testX, testy, nb_classes)[0]
        if score > best_score:
            best_score, best_weights = score, weights
            print('>%s %.3f' % (best_weights, best_score))
    return list(best_weights)


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testy, probabilities_file, nb_classes):
    # make predictions
    predictions = pd.read_csv(probabilities_file)
    # Get the probabilities of each model using pandas
    yhats = [
        convert_str2array(predictions.loc[predictions['path'] == os.path.splitext(model)[0]]['probabilities'].values[0])
        for
        model in members]
    yhats = np.array(yhats)
    yhats = np.reshape(yhats, (len(members), len(testy), nb_classes))

    result = None
    if isinstance(weights, str):
        if weights == "MAXIMUM":
            yhats = np.transpose(yhats, (1, 0, 2))
            yhats = np.reshape(yhats, (len(testy), len(members) * nb_classes))
            result = np.mod(yhats.argmax(axis=-1), nb_classes)
        else:
            print("Weights is %s, Unknown weights variable type.", weights)
    elif isinstance(weights, np.ndarray):
        # weighted sum across ensemble members
        summed = np.tensordot(yhats, weights, axes=(0, 0))
        # argmax across classes
        result = np.argmax(summed, axis=1)
    else:
        print("Unknown weights variable type.")

    return result


# evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, probabilities_file, testy, nb_classes):
    # make prediction
    yhat = ensemble_predictions(members, weights, testy, probabilities_file, nb_classes)
    # calculate accuracy
    return accuracy_score(testy, yhat), yhat

#########################################################################################
## Operations functions
#########################################################################################

def compute_difference_matrices(trained_models_folder,
                                prediction_results_file,
                                results_folder,
                                ensemble_type,
                                ensemble_models_list=None):
    """
    This function returns the computed difference matrices between the ensemble confusion matrices and the individual models matrices.
    Normalization can be applied by setting `normalize=True`.
    """
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    # Determination of test_folds_indices
    test_folds_indices = list(range(0, nb_folds))

    if ensemble_type == "Unique":
        # Prepare the subplots
        figure_multipleplots, axes_subplots = plt.subplots(nrows=nb_folds, ncols=nb_folds - 1, figsize=(35, 35),
                                                           squeeze=True)
        models_name = re.findall("weighted_prediction_results_(\w+).csv", prediction_results_file)[0]
        for test_index in test_folds_indices:
            # Determination of TestSplit from test_index
            test_split = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(test_split, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Determine class labels
            classes = np.arange(0, nb_classes)
            # Target labels
            test_labels = test_data['class'].values
            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)
            # Determination of the validation index
            for val_index in val_folds_indices:
                # Model name
                model_name = models_name + "_split_test" + str(test_index) + "_val" + str(val_index) + "_weights"
                # Ensemble model name
                ensemble_model_name = "Ensemble_" + models_name + "_split_test" + str(test_index)
                # Compute confusion matrix
                prediction_results = pd.read_csv(prediction_results_file)
                y_pred = ast.literal_eval(
                    prediction_results.loc[prediction_results['path'].str.contains(model_name)]['predictions'].values[0])
                confusion_mat = confusion_matrix(test_labels, y_pred)
                confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
                y_pred_ensemble = ast.literal_eval(
                    prediction_results.loc[prediction_results['path'].str.contains(ensemble_model_name)][
                        'predictions'].values[
                        0])
                ensemble_confusion_mat = confusion_matrix(test_labels, y_pred_ensemble)
                ensemble_confusion_mat = ensemble_confusion_mat.astype('float') / ensemble_confusion_mat.sum(axis=1)[:,
                                                                                  np.newaxis]
                difference_confusion_mat = ensemble_confusion_mat - confusion_mat

                fig, ax = plt.subplots()
                im = ax.imshow(difference_confusion_mat, interpolation='nearest', cmap=plt.cm.Reds)
                im.set_clim(-0.1, 0.2)
                val_index = test_index if val_index == nb_folds - 1 else val_index
                im_subplots = axes_subplots[test_index, val_index].imshow(difference_confusion_mat, interpolation='nearest',
                                                                          cmap=plt.cm.Reds)
                # ax.figure.colorbar(im,ax=ax)
                im_subplots.set_clim(-0.1, 0.2)
                # axes_subplots[test_index,val_index].figure.colorbar(im_subplots,ax=axes_subplots[test_index,val_index])
                # We want to show all ticks...
                ax.set(xticks=np.arange(difference_confusion_mat.shape[1]),
                       yticks=np.arange(difference_confusion_mat.shape[0]),
                       # ... and label them with the respective list entries
                       xticklabels=classes, yticklabels=classes,
                       title=model_name,
                       ylabel='True label',
                       xlabel='Predicted label')

                axes_subplots[test_index, val_index].set(xticks=np.arange(difference_confusion_mat.shape[1]),
                                                         yticks=np.arange(difference_confusion_mat.shape[0]),
                                                         # ... and label them with the respective list entries
                                                         xticklabels=classes, yticklabels=classes,
                                                         title=model_name,
                                                         ylabel='True label',
                                                         xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                plt.setp(axes_subplots[test_index, val_index].get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = difference_confusion_mat.max() / 2.
                for i in range(difference_confusion_mat.shape[0]):
                    for j in range(difference_confusion_mat.shape[1]):
                        ax.text(j, i, format(difference_confusion_mat[i, j], fmt),
                                ha="center", va="center",
                                color="white" if difference_confusion_mat[i, j] > thresh else "black")
                        axes_subplots[test_index, val_index].text(j, i, format(difference_confusion_mat[i, j], fmt),
                                                                  ha="center", va="center",
                                                                  color="white" if difference_confusion_mat[
                                                                                       i, j] > thresh else "black")

                fig.tight_layout()
                plt.savefig(os.path.join(results_folder, model_name + '_differenceconfusionmat.pdf'), bbox_inches='tight')

        figure_multipleplots.tight_layout()
        figure_multipleplots.savefig(os.path.join(results_folder, models_name + "_differenceconfusionmatrix_subplots.pdf"),
                                     bbox_inches='tight', pad_inches=0)

        # plt.show()
    else: #ensemble_type == "Global"
        # Prepare the subplots
        nb_ensemble_models = len(ensemble_models_list)
        figure_multipleplots, axes_subplots = plt.subplots(nrows=nb_ensemble_models, ncols=nb_folds, figsize=(35, 35),
                                                           squeeze=True)
        models_names = re.findall("global_ensemble_summed_prediction_results_\w+_folds_(\w+)_.csv", prediction_results_file)[0]
        for test_index in test_folds_indices:
            # Determination of TestSplit from test_index
            test_split = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(test_split, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Determine class labels
            classes = np.arange(0, nb_classes)
            # Target labels
            test_labels = test_data['class'].values
            # Global Ensemble Model name
            global_ensemble_model_name = "Global_Ensemble_" + models_names + "_split_test" + str(test_index)
            # Compute confusion matrix
            prediction_results = pd.read_csv(prediction_results_file)
            print(prediction_results_file)
            print(global_ensemble_model_name)
            y_pred = ast.literal_eval(
                prediction_results.loc[prediction_results['path'].str.contains(global_ensemble_model_name)]['predictions'].values[
                    0])
            global_ensemble_confusion_mat = confusion_matrix(test_labels, y_pred)
            global_ensemble_confusion_mat = global_ensemble_confusion_mat.astype('float') / global_ensemble_confusion_mat.sum(axis=1)[:, np.newaxis]

            ensemble_model_index = 0
            for ensemble_model in ensemble_models_list:
                # Get individual ensemble model name
                if ensemble_model == "TWOSTREAM_I3D_PRETRAINED_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3":
                    ensemble_model = "TWOSTREAM_I3D_PRETRAINED_WITH_DA"
                    ensemble_model_name = "Ensemble_" + str(nb_folds) + "folds_" \
                                          + "TWOSTREAM_I3D_PRETRAINED_CS_unbalanced_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3" \
                                          + "_split_test" + str(test_index)
                    # Find individual prediction_results_file
                    individual_ensemble_prediction_results_file = lookFor_UniqueEnsemble_predictionsFile(nb_folds,
                                                                                     results_folder,
                                                                                     "TWOSTREAM_I3D",
                                                                                     "_PRETRAINED",
                                                                                     "unbalanced",
                                                                                     "FarneBack_onTheFly",
                                                                                     "augmented_precomputed",
                                                                                     3)
                else:
                    ensemble_model_name = "Ensemble_" + str(nb_folds) + "folds_" + ensemble_model \
                                          + "_CS_unbalanced_OF_TVL1_precomputed_AS_non_augmented_split_test" \
                                          + str(test_index)

                    model_type, training_condition = getModelTypeAndTrainingCondition(ensemble_model)
                    individual_ensemble_prediction_results_file = lookFor_UniqueEnsemble_predictionsFile(nb_folds,
                                                                                     results_folder,
                                                                                     model_type,
                                                                                     training_condition,
                                                                                     "unbalanced",
                                                                                     "TVL1_precomputed",
                                                                                     "non_augmented",
                                                                                     0)
                # Compute individual ensemble model confusion matrix
                print(ensemble_model_name)
                prediction_results = pd.read_csv(individual_ensemble_prediction_results_file)
                y_pred = ast.literal_eval(
                    prediction_results.loc[prediction_results['path'].str.contains(ensemble_model_name)]['predictions'].values[
                        0])
                ensemble_confusion_mat = confusion_matrix(test_labels, y_pred)
                ensemble_confusion_mat = ensemble_confusion_mat.astype('float') / ensemble_confusion_mat.sum(axis=1)[:, np.newaxis]
                difference_confusion_mat = global_ensemble_confusion_mat - ensemble_confusion_mat

                fig, ax = plt.subplots()
                im = ax.imshow(difference_confusion_mat, interpolation='nearest', cmap=plt.cm.Reds)
                im.set_clim(-0.1, 0.2)
                im_subplots = axes_subplots[ensemble_model_index, test_index].imshow(difference_confusion_mat,
                                                                          interpolation='nearest',
                                                                          cmap=plt.cm.Reds)
                # ax.figure.colorbar(im,ax=ax)
                im_subplots.set_clim(-0.1, 0.2)
                # axes_subplots[test_index,val_index].figure.colorbar(im_subplots,ax=axes_subplots[test_index,val_index])
                # We want to show all ticks...
                ax.set(xticks=np.arange(difference_confusion_mat.shape[1]),
                       yticks=np.arange(difference_confusion_mat.shape[0]),
                       # ... and label them with the respective list entries
                       xticklabels=classes, yticklabels=classes,
                       # title="Diff_conf_Global-Ensemble_"+ensemble_model+"_split_test"+str(test_index),
                       title="",
                       ylabel='True label',
                       xlabel='Predicted label')

                axes_subplots[ensemble_model_index, test_index].set(xticks=np.arange(difference_confusion_mat.shape[1]),
                                                         yticks=np.arange(difference_confusion_mat.shape[0]),
                                                         # ... and label them with the respective list entries
                                                         xticklabels=classes, yticklabels=classes,
                                                         # title="Diff_conf_Global-Ensemble_"+ensemble_model+"_split_test"+str(test_index),
                                                         title="",
                                                         ylabel='True label',
                                                         xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                plt.setp(axes_subplots[ensemble_model_index, test_index].get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = difference_confusion_mat.max() / 2.
                for i in range(difference_confusion_mat.shape[0]):
                    for j in range(difference_confusion_mat.shape[1]):
                        ax.text(j, i, format(difference_confusion_mat[i, j], fmt),
                                ha="center", va="center",
                                color="white" if difference_confusion_mat[i, j] > thresh else "black")
                        axes_subplots[ensemble_model_index, test_index].text(j, i, format(difference_confusion_mat[i, j], fmt),
                                                                  ha="center", va="center",
                                                                  color="white" if difference_confusion_mat[
                                                                                       i, j] > thresh else "black")

                fig.tight_layout()
                plt.savefig(os.path.join(results_folder, ensemble_model + "_split_test" + str(test_index) + '_differenceconfusionmat.pdf'),
                            bbox_inches='tight')
                ensemble_model_index = ensemble_model_index + 1

        figure_multipleplots.tight_layout()
        figure_multipleplots.savefig(os.path.join(results_folder, models_names + "_differenceconfusionmatrix_subplots.pdf"),
                                     bbox_inches='tight', pad_inches=0)



def compute_confusion_matrices(trained_models_folder,
                               prediction_results_file,
                               ensemble_type,
                               results_folder):
    """
    This function returns the computed confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param trained_models_folder:
    :param prediction_results_file:
    :param ensemble_type: Unique or Global ensemble
    :param results_folder:
    :return:
    """
    # Prepare the subplots
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    figure_multipleplots, axes_subplots = plt.subplots(nrows=nb_folds, ncols=nb_folds, figsize=(35, 35), squeeze=True)

    # Determination of test_indices from nb_folds
    test_folds_indices = list(range(0, nb_folds))

    if ensemble_type == "Unique":
        models_name = re.findall("weighted_prediction_results_(\w+).csv", prediction_results_file)[0]
        for test_index in test_folds_indices:
            # Determination of TestSplit from test_index
            test_split = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(test_split, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Determine class labels
            classes = np.arange(0, nb_classes)
            # Target labels
            test_labels = test_data['class'].values
            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)
            # Determination of the validation index
            for val_index in val_folds_indices:
                # Model name
                model_name = models_name + "_split_test" + str(test_index) + "_val" + str(val_index) + "_weights"

                # Compute confusion matrix
                prediction_results = pd.read_csv(prediction_results_file)
                y_pred = ast.literal_eval(
                    prediction_results.loc[prediction_results['path'].str.contains(model_name)]['predictions'].values[0])
                confusion_mat = confusion_matrix(test_labels, y_pred)
                confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

                fig, ax = plt.subplots()
                # im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
                # im_subplots = axes_subplots[test_index, val_index].imshow(confusion_mat, interpolation='nearest',
                #                                                           cmap=plt.cm.Blues)
                # ax.figure.colorbar(im,ax=ax)
                # axes_subplots[test_index,val_index].figure.colorbar(im_subplots,ax=axes_subplots[test_index,val_index])
                # We want to show all ticks...
                ax.set(xticks=np.arange(confusion_mat.shape[1]),
                       yticks=np.arange(confusion_mat.shape[0]),
                       # ... and label them with the respective list entries
                       xticklabels=classes, yticklabels=classes,
                       title=model_name,
                       ylabel='True label',
                       xlabel='Predicted label')

                axes_subplots[test_index, val_index].set(xticks=np.arange(confusion_mat.shape[1]),
                                                         yticks=np.arange(confusion_mat.shape[0]),
                                                         # ... and label them with the respective list entries
                                                         xticklabels=classes, yticklabels=classes,
                                                         title=model_name,
                                                         ylabel='True label',
                                                         xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                plt.setp(axes_subplots[test_index, val_index].get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = confusion_mat.max() / 2.
                for i in range(confusion_mat.shape[0]):
                    for j in range(confusion_mat.shape[1]):
                        ax.text(j, i, format(confusion_mat[i, j], fmt),
                                ha="center", va="center",
                                color="white" if confusion_mat[i, j] > thresh else "black")
                        axes_subplots[test_index, val_index].text(j, i, format(confusion_mat[i, j], fmt),
                                                                  ha="center", va="center",
                                                                  color="white" if confusion_mat[
                                                                                       i, j] > thresh else "black")

                fig.tight_layout()
                plt.savefig(os.path.join(results_folder, model_name + '_confusionmatrix.pdf'), bbox_inches='tight')

        # Ensemble models confusion matrices
        val_index = -1
        for test_index in test_folds_indices:
            # Find a valid test split
            test_split = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(test_split, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Determine class labels
            classes = np.arange(0, nb_classes)
            # Target labels
            test_labels = test_data['class'].values
            # Model name
            model_name = "Ensemble_" + models_name + "_split_test" + str(test_index)
            # Compute confusion matrix
            prediction_results = pd.read_csv(prediction_results_file)
            y_pred = ast.literal_eval(
                prediction_results.loc[prediction_results['path'].str.contains(model_name)]['predictions'].values[0])
            confusion_mat = confusion_matrix(test_labels, y_pred)
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots()
            im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
            im_subplots = axes_subplots[test_index, test_index].imshow(confusion_mat, interpolation='nearest',
                                                                       cmap=plt.cm.Blues)
            # ax.figure.colorbar(im,ax=ax)
            # axes_subplots[test_index, test_index].figure.colorbar(im_subplots,ax=axes_subplots[test_index, val_index])
            # We want to show all ticks...
            ax.set(xticks=np.arange(confusion_mat.shape[1]),
                   yticks=np.arange(confusion_mat.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title="Ensemble_split_test"+str(test_index),
                   ylabel='True label',
                   xlabel='Predicted label')

            axes_subplots[test_index, test_index].set(xticks=np.arange(confusion_mat.shape[1]),
                                                      yticks=np.arange(confusion_mat.shape[0]),
                                                      # ... and label them with the respective list entries
                                                      xticklabels=classes, yticklabels=classes,
                                                      title="Ensemble_split_test"+str(test_index),
                                                      ylabel='True label',
                                                      xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            plt.setp(axes_subplots[test_index, test_index].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            # Loop over data dimensions and create text annotations.
            fmt = '.2f'
            thresh = confusion_mat.max() / 2.
            for i in range(confusion_mat.shape[0]):
                for j in range(confusion_mat.shape[1]):
                    ax.text(j, i, format(confusion_mat[i, j], fmt),
                            ha="center", va="center",
                            color="white" if confusion_mat[i, j] > thresh else "black")
                    axes_subplots[test_index, test_index].text(j, i, format(confusion_mat[i, j], fmt),
                                                               ha="center", va="center",
                                                               color="white" if confusion_mat[i, j] > thresh else "black")

            fig.tight_layout()
            plt.savefig(os.path.join(results_folder, model_name + "_confusionmatrix.pdf"), bbox_inches="tight")

        figure_multipleplots.tight_layout()
        figure_multipleplots.savefig(os.path.join(results_folder, models_name + "_confusionmatrix_subplots.pdf"),
                                     bbox_inches='tight')

        # plt.show()
    else: #ensemble_type == "Global"
        models_names = re.findall("global_ensemble_summed_prediction_results_\w+_folds_(\w+)_.csv", prediction_results_file)[0]
        for test_index in test_folds_indices:
            # Find a valid test split
            test_split = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(test_split, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Determine class labels
            classes = np.arange(0, nb_classes)
            # Target labels
            test_labels = test_data['class'].values
            # Model name
            model_name = "Global_Ensemble_" + models_names + "_split_test" + str(test_index)
            # Compute confusion matrix
            prediction_results = pd.read_csv(prediction_results_file)
            y_pred = ast.literal_eval(
                prediction_results.loc[prediction_results['path'].str.contains(model_name)]['predictions'].values[0])
            confusion_mat = confusion_matrix(test_labels, y_pred)
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots()
            im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
            im_subplots = axes_subplots[test_index, test_index].imshow(confusion_mat, interpolation='nearest',
                                                                       cmap=plt.cm.Blues)
            # ax.figure.colorbar(im,ax=ax)
            # axes_subplots[test_index, test_index].figure.colorbar(im_subplots,ax=axes_subplots[test_index, val_index])
            # We want to show all ticks...
            ax.set(xticks=np.arange(confusion_mat.shape[1]),
                   yticks=np.arange(confusion_mat.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title="Global_ensemble_split_test"+str(test_index),
                   ylabel='True label',
                   xlabel='Predicted label')

            axes_subplots[test_index, test_index].set(xticks=np.arange(confusion_mat.shape[1]),
                                                      yticks=np.arange(confusion_mat.shape[0]),
                                                      # ... and label them with the respective list entries
                                                      xticklabels=classes, yticklabels=classes,
                                                      title="Global_ensemble_split_test"+str(test_index),
                                                      ylabel='True label',
                                                      xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            plt.setp(axes_subplots[test_index, test_index].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            # Loop over data dimensions and create text annotations.
            fmt = '.2f'
            thresh = confusion_mat.max() / 2.
            for i in range(confusion_mat.shape[0]):
                for j in range(confusion_mat.shape[1]):
                    ax.text(j, i, format(confusion_mat[i, j], fmt),
                            ha="center", va="center",
                            color="white" if confusion_mat[i, j] > thresh else "black")
                    axes_subplots[test_index, test_index].text(j, i, format(confusion_mat[i, j], fmt),
                                                               ha="center", va="center",
                                                               color="white" if confusion_mat[i, j] > thresh else "black")

            fig.tight_layout()
            plt.savefig(os.path.join(results_folder, model_name + "_confusionmatrix.pdf"), bbox_inches="tight")

        figure_multipleplots.tight_layout()
        figure_multipleplots.savefig(os.path.join(results_folder, models_names + "_confusionmatrix_subplots.pdf"),
                                     bbox_inches='tight')





def stickDiagrams_wellClassifiedClips_per_numberOfModels(trained_models_folder,
                                                         probabilities_file,
                                                         involved_sets,
                                                         models_name,
                                                         results_folder):
    """
    Count the number of the maximum networks that retrieved the correct class of a video clip
    """
    # Plot Arguments
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    labels = [str(num) for num in range(0, nb_folds)]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nombre de clips du test bien classés')
    ax.set_xlabel('Nombre de modèles individuels qui ont bien classé un clip du test')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    # Arguments
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    test_folds_indices = list(range(0, nb_folds))
    # List used to store the prediction results of each model
    correctly_classified_video_counter_list = list()
    predictions = pd.read_csv(probabilities_file)

    # Creation of the results output repository
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if involved_sets == "test":
        # Determination of the test index
        for test_index in test_folds_indices:
            # Determinate the test folder
            data_folder = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Predict labels from video clips
            test_labels = test_data['class'].values

            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)

            trained_model_paths = list()
            for val_index in val_folds_indices:
                model_file = models_name + "_split_test" + str(test_index) + "_val" + str(
                    val_index) + "_weights"
                trained_model_path = os.path.join(data_folder, model_file)
                trained_model_paths.append(trained_model_path)

            # Count the number of correctly predicted labels of test video clips
            probabilities = [
                convert_str2array(
                    predictions.loc[predictions['path'] == model]['probabilities'].values[0])
                for model in trained_model_paths]
            probabilities = np.array(probabilities)
            probabilities = np.reshape(probabilities, (len(trained_model_paths), len(test_labels), nb_classes))
            probabilities = np.transpose(probabilities, (1, 0, 2))
            probabilities = np.reshape(probabilities, (len(test_labels), len(trained_model_paths), nb_classes))
            predicted_videos = probabilities.argmax(axis=-1)
            correctly_predicted_videos = predicted_videos.T == test_labels
            correctly_classified_video_counter = np.count_nonzero(correctly_predicted_videos == True, axis=0)

            counter = [
                np.count_nonzero(correctly_classified_video_counter == 0),
                np.count_nonzero(correctly_classified_video_counter == 1),
                np.count_nonzero(correctly_classified_video_counter == 2),
                np.count_nonzero(correctly_classified_video_counter == 3),
                np.count_nonzero(correctly_classified_video_counter == 4)
            ]

            correctly_classified_video_counter_list.append(counter)

    else:  # involved_sets == "train_val"
        # Determination of the test index
        for test_index in test_folds_indices:
            # Determinate the test folder
            data_folder = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'val.csv'))
            train_val_data = train_data.append(val_data, ignore_index=True)
            # Determine the number of classes
            nb_classes = len(set(train_val_data['class']))
            # Predict labels from video clips
            trainval_labels = train_val_data['class'].values

            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)

            trained_model_paths = list()
            for val_index in val_folds_indices:
                model_file = models_name + "_split_test" + str(test_index) + "_val" + str(
                    val_index) + "_weights"
                trained_model_path = os.path.join(data_folder, model_file)
                trained_model_paths.append(trained_model_path)

            # Count the number of correctly predicted labels of test video clips
            probabilities = [
                convert_str2array(
                    predictions.loc[predictions['path'] == model]['probabilities'].values[0])
                for model in trained_model_paths]
            probabilities = np.array(probabilities)
            probabilities = np.reshape(probabilities, (len(trained_model_paths), len(trainval_labels), nb_classes))
            probabilities = np.transpose(probabilities, (1, 0, 2))
            probabilities = np.reshape(probabilities, (len(trainval_labels), len(trained_model_paths), nb_classes))
            predicted_videos = probabilities.argmax(axis=-1)
            correctly_predicted_videos = predicted_videos.T == trainval_labels
            correctly_classified_video_counter = np.count_nonzero(correctly_predicted_videos == True, axis=0)

            counter = [
                np.count_nonzero(correctly_classified_video_counter == 0),
                np.count_nonzero(correctly_classified_video_counter == 1),
                np.count_nonzero(correctly_classified_video_counter == 2),
                np.count_nonzero(correctly_classified_video_counter == 3),
                np.count_nonzero(correctly_classified_video_counter == 4)
            ]
            correctly_classified_video_counter_list.append(counter)

    # Unique stick diagram
    ax.bar(x - 2 * width, correctly_classified_video_counter_list[0], width,
           label="Ensemble associé à l'échantillon de test " + str(0))
    ax.bar(x - width, correctly_classified_video_counter_list[1], width,
           label="Ensemble associé à l'échantillon de test " + str(1))
    ax.bar(x - width / 100, correctly_classified_video_counter_list[2], width,
           label="Ensemble associé à l'échantillon de test " + str(2))
    ax.bar(x + 0.95 * width, correctly_classified_video_counter_list[3], width,
           label="Ensemble associé à l'échantillon de test " + str(3))
    ax.bar(x + 1.95 * width, correctly_classified_video_counter_list[4], width,
           label="Ensemble associé à l'échantillon de test " + str(4))
    ax.legend()
    fig.tight_layout()

    plt.savefig(os.path.join(results_folder,
                             'Counting_the_number_of_' + involved_sets + '_clips_retrieved_byAmax_numberofNetworks_stick_diagram_' + models_name + ".pdf"),
                bbox_inches='tight')
    plt.show()


def store_probabilities(trained_models_folder, results_folder, involved_sets, batch_size, workers, model_type,
                        training_condition, optical_flow_status, augmentation_status, augmentation_frequency,
                        classes_status, models_name):
    """
    :param models_name: str, mention the models stem name
    Predicts the models probabilities for the train_val or the test set.
    Stores the probabilities in an output csv file.
    """
    # Arguments
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    test_folds_indices = list(range(0, nb_folds))
    # List used to store the prediction results of each model
    store_models_probabilities = list()

    # Creation of the results output repository
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if involved_sets == "test":
        # Determination of the test index
        for test_index in test_folds_indices:
            # Determinate the test folder  
            data_folder = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
            # Determine the number of classes
            nb_classes = len(set(test_data['class']))
            # Define input
            sample_input = define_input(model_type)
            # Create the test generator
            video_test_generator = DataGenerator(test_data,
                                                 model_type,
                                                 sample_input.shape,
                                                 nb_classes,
                                                 batch_size=1,
                                                 optical_flow_status=optical_flow_status,
                                                 augmentation_status="non_augmented",
                                                 augmentation_frequency=0,
                                                 shuffle=False)

            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)
            # Determine the multiprocessing status
            use_multiprocessing = True
            # Predict a model associated to each validation fold
            for val_index in val_folds_indices:
                model_file = models_name + "_split_test" + str(test_index) + "_val" + str(val_index) + "_weights.hdf5"
                trained_model_path = os.path.join(data_folder, model_file)
                model = evaluate_load_model(model_type, trained_model_path, sample_input.shape, nb_classes)
                model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
                local_predictions = model.predict_generator(video_test_generator,
                                                            workers=workers,
                                                            use_multiprocessing=use_multiprocessing,
                                                            verbose=1)
                print(local_predictions.shape)
                store_models_probabilities.append(
                    [os.path.splitext(trained_model_path)[0], convert_array2listofarrays(local_predictions)])
        # Save results in a csv file
        csv_file_path = os.path.join(results_folder, "test_predicted_probabilities_" + models_name + ".csv")
        predicted_probabilities = pd.DataFrame(store_models_probabilities, columns=["path", "probabilities"])
        predicted_probabilities.to_csv(csv_file_path)
        return csv_file_path
    else:  # Train_val
        # Determination of the test index
        for test_index in test_folds_indices:
            # Determinate the test folder
            data_folder = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
            # Load test data
            trainval_data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'val.csv'))
            trainval_data = trainval_data.append(val_data, ignore_index=True)
            # Determine the number of classes
            nb_classes = len(set(trainval_data['class']))
            # Define input
            sample_input = define_input(model_type)
            # Create the train_val generator
            video_trainval_generator = DataGenerator(trainval_data,
                                                     model_type,
                                                     sample_input.shape,
                                                     nb_classes,
                                                     batch_size=1,
                                                     optical_flow_status=optical_flow_status,
                                                     augmentation_status="non_augmented",
                                                     augmentation_frequency=0,
                                                     shuffle=False)
            # Create the possible validation folds to select by excluding the selected test index
            val_folds_indices = test_folds_indices.copy()
            val_folds_indices.remove(test_index)
            # Determine the multiprocessing status
            use_multiprocessing = True
            # Predict a model associated to each validation fold
            for val_index in val_folds_indices:
                model_file = models_name + "_split_test" + str(test_index) + "_val" + str(val_index) + "_weights.hdf5"
                trained_model_path = os.path.join(data_folder, model_file)
                model = evaluate_load_model(model_type, trained_model_path, sample_input.shape, nb_classes)
                model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
                local_predictions = model.predict_generator(video_trainval_generator,
                                                            workers=workers,
                                                            use_multiprocessing=use_multiprocessing,
                                                            verbose=1)
                store_models_probabilities.append(
                    [os.path.splitext(trained_model_path)[0], convert_array2listofarrays(local_predictions)])
        # Save results in a csv file
        csv_file_path = os.path.join(results_folder, "train_val_predicted_probabilities_" + models_name + ".csv")
        predicted_probabilities = pd.DataFrame(store_models_probabilities, columns=["path", "probabilities"])
        predicted_probabilities.to_csv(csv_file_path)
        return csv_file_path


def evaluate_ensembles(trained_models_folder,
                       results_folder,
                       weights_type,
                       histories_folder,
                       test_probabilities_file,
                       trainval_probabilities_file,
                       weights_array_file,
                       batch_size,
                       workers,
                       model_type,
                       training_condition,
                       optical_flow_status,
                       augmentation_status,
                       augmentation_frequency,
                       classes_status,
                       models_name):
    """
    Evaluate ensembles. The function either calculates the probabilities by calling store_probabilities or get the probabilities as an argument.
    :param trained_models_folder : str, folder where the trained models are stored. The folder name is also processed to guess the model_type, training_condition, augmentation_status, optical_flow_status.
    :param results_folder : str, folder where the probabilities_file and the predictions_file are stored
    :param weights_type : str, type of the weights used to combine the models within an Ensemble
    :param histories_folder : str, the parent folder that contains the validation losses of each model.
    :param test_probabilities_file : str, path to the probabilities precomputed for the test file
    :param trainval_probabilities_file : str, path to the probabilities precomputed for the train_val files (needed to compute the weights for grid search and differential evolution)
    :param weights_array_file : str, path to the weights_array_file if pre_computed
    :param batch_size : int, mention the batch size for the predict_generator function
    :param workers : int, mention the number of workers for the predict_generator function
    :param model_type : str, mention the model architecture type
    :param training_condition : str, mention in which condition the models were trained
    :param optical_flow_status : str, mention the optical flow status (either FarneBack computed on the fly or TVL1 precomputed)
    :param augmentation_status : str, mention whether the augmentation was performed on the fly, precomputed, or not done at all
    :param augmentation_frequency : int, says how often the data was augmented
    :param classes_status : str, says if the models were trained on a balanced dataset or not
    :param models_name: str, mention the models stem name
    """
    ## Arguments
    nb_folds = int(os.path.basename(trained_models_folder)[0])
    test_folds_indices = list(range(0, nb_folds))

    # List used to store the prediction results of each model
    store_models_predictions = list()

    # Lists used to store the Grid_Search and Differential_Evolution weights for all the ensembles
    optimization_weights = list()

    # Creation of the results output repository
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    # Load probabilities
    if test_probabilities_file == None:
        involved_sets = "test"
        test_probabilities_file = store_probabilities(trained_models_folder,
                                                      results_folder,
                                                      involved_sets,
                                                      batch_size,
                                                      workers,
                                                      model_type,
                                                      training_condition,
                                                      optical_flow_status,
                                                      augmentation_status,
                                                      augmentation_frequency,
                                                      classes_status,
                                                      models_name)
    ## Evaluate ensembles according to the weighting method
    # Determination of the test index
    for test_index in test_folds_indices:
        # Determinate the test folder
        data_folder = os.path.join(trained_models_folder, "TestSplit" + str(test_index))
        # Load train_val and test data
        test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        trainval_data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
        val_data = pd.read_csv(os.path.join(data_folder, 'val.csv'))
        trainval_data = trainval_data.append(val_data, ignore_index=True)
        # Determine the number of classes
        nb_classes = len(set(test_data['class']))
        # Find labels from video clips
        test_labels = test_data['class'].values
        trainval_labels = trainval_data['class'].values
        # Create the possible validation folds to select by excluding the selected test index
        val_folds_indices = test_folds_indices.copy()
        val_folds_indices.remove(test_index)
        # Get the trained model names
        trained_model_paths = list()
        for val_index in val_folds_indices:
            model_file = models_name + "_split_test" + str(test_index) + "_val" + str(val_index) + "_weights"
            trained_model_path = os.path.join(data_folder, model_file)
            accuracy_score, single_model_predictions = evaluate_single_model(trained_model_path, test_labels,
                                                                             test_probabilities_file, nb_classes)
            print("Model val %d : %f" % (val_index, accuracy_score))
            store_models_predictions.append([trained_model_path, convert_array2listofarrays(single_model_predictions)])
            trained_model_paths.append(trained_model_path)
        # Compute/retrieve the weights according to the weighting method
        ensemble_models_number = nb_folds - 1
        weights = None
        if weights_type == "GRID_SEARCH":
            if trainval_probabilities_file == None:
                involved_sets = "train_val"
                trainval_probabilities_file = store_probabilities(trained_models_folder,
                                                                  results_folder,
                                                                  involved_sets,
                                                                  batch_size,
                                                                  workers,
                                                                  model_type,
                                                                  training_condition,
                                                                  optical_flow_status,
                                                                  augmentation_status,
                                                                  augmentation_frequency,
                                                                  classes_status,
                                                                  models_name)
            if weights_array_file == None:
                optimization_weights.append(
                    apply_grid_search(trained_model_paths, trainval_probabilities_file, trainval_labels))
                weights = optimization_weights[test_index]
            else:
                weights = np.load(weights_array_file)[test_index]
        elif weights_type == "DIFFERENTIAL_EVOLUTION":
            if trainval_probabilities_file == None:
                involved_sets = "train_val"
                trainval_probabilities_file = store_probabilities(trained_models_folder,
                                                                  results_folder,
                                                                  involved_sets,
                                                                  batch_size,
                                                                  workers,
                                                                  model_type,
                                                                  training_condition,
                                                                  optical_flow_status,
                                                                  augmentation_status,
                                                                  augmentation_frequency,
                                                                  classes_status,
                                                                  models_name)

            if weights_array_file == None:
                optimization_weights.append(apply_differential_evolution(ensemble_models_number, trained_model_paths,
                                                                         trainval_probabilities_file, trainval_labels))
                weights = optimization_weights[test_index]
            else:
                weights = np.load(weights_array_file)[test_index]
        elif weights_type == "SUM":
            weights = np.ones(ensemble_models_number)
        elif weights_type == "VALIDATION_ERROR_INVERSE":
            weights = get_modeltraining_validation_loss(histories_folder, test_index)
        elif weights_type == "MAXIMUM":
            weights = weights_type
        else:
            print("Unknown weighting method.")
        # Predict predictions for ensembles according to the choosen weighting method
        ensemble_model_accuracy, ensemble_model_predictions = evaluate_ensemble(trained_model_paths, weights,
                                                                                test_probabilities_file, test_labels,
                                                                                nb_classes)
        print("Fold %d : %f" % (test_index, ensemble_model_accuracy))
        ensemble_model_name = "Ensemble_" + models_name + "_split_test" + str(test_index)
        store_models_predictions.append([ensemble_model_name, convert_array2listofarrays(ensemble_model_predictions)])

    # Save results in a csv file
    csv_file_path = os.path.join(results_folder, "weighted_prediction_results_" + models_name + ".csv")
    predictions_results = pd.DataFrame(store_models_predictions, columns=["path", "predictions"])
    predictions_results.to_csv(csv_file_path)
    # Save optimization weights
    if weights_type == "GRID_SEARCH" or weights_type == "DIFFERENTIAL_EVOLUTION":
        optimization_weights_filename = weights_type + "_" + models_name + ".npy"
        np.save(optimization_weights_filename, np.array(optimization_weights))
    return (csv_file_path)


#########################################################################################
## Global model evaluation
#########################################################################################

def compute_combinations(models_list):
    """
    We compute combinations without replacement of the models' list when we create tuples of 1 to 8 models
    :param models_list: list of the models from which we compute the combinations
    :return: the number of combinations and the list of the combinations
    """
    nb_combinations = 0
    combinations = list()

    for picked_models_quantity in range(1, len(models_list) + 1):
        current_combinations = list(set(list(itertools.combinations(models_list, picked_models_quantity))))
        nb_combinations = nb_combinations + len(current_combinations)
        combinations.append(current_combinations)

    combinations = list(itertools.chain.from_iterable(combinations))
    return nb_combinations, combinations


def combine_ensembles(nb_folds, trained_models_parent_folder, models_list, results_folder):
    """
    Evaluate combinations of ensembles
    :param nb_folds: number of folds
    :param trained_models_parent_folder: folder containing all the trained_models
    :param models_list: list of model names mentioning the architecture and the training condition
    "Architecture_TrainingCondition". Example : C3D_PRETRAINED, TWOSTREAM_I3D_SCRATCH
    :param results_folder: Default (Results/)
    :return: predictions resulting from the models global combination
    """

    # Arguments
    # Compute combinations
    nb_combinations, combinations = compute_combinations(models_list)
    global_ensemble_accuracy = dict()

    # Evaluate each combination giving birth to a global ensemble
    for combination in combinations:
        global_ensemble_accuracy[combination] = global_evaluate_ensembles(nb_folds,
                                                                          trained_models_parent_folder,
                                                                          combination,
                                                                          results_folder)

    # Sort the combinations according to the accuracy values
    ordered_global_ensemble_accuracy = dict(sorted(global_ensemble_accuracy.items(), key=lambda item: item[1], reverse=True))

    # Display the resulting combinations accuracies
    for combination, accuracy in ordered_global_ensemble_accuracy.items():
        print(combination, accuracy)


def global_evaluate_ensembles(nb_folds, trained_models_parent_folder, models_list, results_folder):
    """
    Global evaluate ensembles of models
    :param nb_folds: number of folds
    :param trained_models_parent_folder: folder containing all the trained_models
    :param models_list: list of model names mentioning the architecture and the training condition
    "Architecture_TrainingCondition". Example : C3D_PRETRAINED, TWOSTREAM_I3D_SCRATCH
    :param results_folder: Default (Results/)
    :return: average accuracy of the global ensemble
    """

    # Arguments
    test_folds_indices = list(range(0, nb_folds))

    # List used to store the prediction results of each model
    store_models_predictions = list()
    store_models_accuracies = list()
    # Creation of the results output repository
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # Models_training dictionary that associates a model_type to a list of training conditions
    # Example : {"C3D" : ["_PRETRAINED", "_SCRATCH"], "I3D" : ["_PRETRAINED"]}
    model_trainingConditions = createModelsTrainingConditionsDictionary(models_list)
    print(model_trainingConditions)
    all_models_names_string = ""
    # Determination of the test index
    for test_index in test_folds_indices:
        all_models_names = list()
        # Array probabilities
        all_models_test_probabilities = pd.DataFrame(columns=["path", "probabilities"])
        trained_model_paths = list()
        # Load probabilities
        ensemble_models_number = 0
        for model_type in model_trainingConditions.keys():
            for training_condition in model_trainingConditions[model_type]:
                if model_type + training_condition == "SPECIALCASE_PRETRAINED":
                    all_models_names.append(
                        "TWOSTREAM_I3D_PRETRAINED_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3")
                    model_type = "TWOSTREAM_I3D"
                    training_condition = "_PRETRAINED"
                    test_probabilities_file = lookFor_probabilitiesFile(nb_folds,
                                                                        results_folder,
                                                                        model_type,
                                                                        training_condition,
                                                                        classes_status="unbalanced",
                                                                        optical_flow_status="FarneBack_onTheFly",
                                                                        augmentation_status="augmented_precomputed",
                                                                        augmentation_frequency=3,
                                                                        involved_sets="test")
                    models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(nb_folds,
                                                                                                    trained_models_parent_folder,
                                                                                                    model_type,
                                                                                                    training_condition,
                                                                                                    classes_status="unbalanced",
                                                                                                    optical_flow_status="FarneBack_onTheFly",
                                                                                                    augmentation_status="augmented_precomputed",
                                                                                                    augmentation_frequency=3)
                else:
                    all_models_names.append(model_type + training_condition)
                    test_probabilities_file = lookFor_probabilitiesFile(nb_folds,
                                                                        results_folder,
                                                                        model_type,
                                                                        training_condition,
                                                                        classes_status="unbalanced",
                                                                        optical_flow_status="TVL1_precomputed",
                                                                        augmentation_status="non_augmented",
                                                                        augmentation_frequency=0,
                                                                        involved_sets="test")
                    models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(nb_folds,
                                                                                                    trained_models_parent_folder,
                                                                                                    model_type,
                                                                                                    training_condition,
                                                                                                    classes_status="unbalanced",
                                                                                                    optical_flow_status="TVL1_precomputed",
                                                                                                    augmentation_status="non_augmented",
                                                                                                    augmentation_frequency=0)
                if test_probabilities_file == None:
                    involved_sets = "test"
                    test_probabilities = pd.read_csv(store_probabilities(trained_models_subfolder,
                                                                         results_folder,
                                                                         involved_sets,
                                                                         batch_size=1,
                                                                         workers=1,
                                                                         model_type=model_type,
                                                                         training_condition=training_condition,
                                                                         optical_flow_status="TVL1_precomputed",
                                                                         augmentation_status="non_augmented",
                                                                         augmentation_frequency=0,
                                                                         classes_status="unbalanced",
                                                                         models_name=models_name))
                else:
                    test_probabilities = pd.read_csv(test_probabilities_file)
                all_models_test_probabilities = all_models_test_probabilities.append(test_probabilities,
                                                                                     sort=False,
                                                                                     ignore_index=True)
                ensemble_models_number = ensemble_models_number + 1
                ## Evaluate ensembles according to the weighting method
                # Determinate the test folder
                data_folder = os.path.join(trained_models_subfolder, "TestSplit" + str(test_index))
                # Create the possible validation folds to select by excluding the selected test index
                val_folds_indices = test_folds_indices.copy()
                val_folds_indices.remove(test_index)
                # Get the trained model names
                for val_index in val_folds_indices:
                    model_file = models_name + "_split_test" + str(test_index) + "_val" + str(
                        val_index) + "_weights"
                    trained_model_path = os.path.join(data_folder, model_file)
                    trained_model_paths.append(trained_model_path)

        separator = '_'
        all_models_names_string = separator.join(all_models_names)
        all_models_test_probabilities_file = "global_ensemble_probabilities_" + all_models_names_string + \
                                             "_TestFold" + str(test_index) + "_" + str(nb_folds) + "folds.csv"
        all_models_test_probabilities.to_csv(os.path.join(results_folder, all_models_test_probabilities_file))

        # Get an example of trained_models_subfolder to obtain the paths to the TestSplits folders
        first_models_folder = os.path.dirname(os.path.dirname(trained_model_paths[0]))
        data_folder = os.path.join(first_models_folder, "TestSplit" + str(test_index))
        # Load train_val and test data
        test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        # Determine the number of classes
        nb_classes = len(set(test_data['class']))
        # Find labels from video clips
        test_labels = test_data['class'].values
        # Predict predictions for ensembles according to the chosen weighting method
        weights = np.ones(ensemble_models_number * (nb_folds - 1))
        ensemble_model_accuracy, ensemble_model_predictions = evaluate_ensemble(trained_model_paths,
                                                                                weights,
                                                                                os.path.join(results_folder,
                                                                                             all_models_test_probabilities_file),
                                                                                test_labels,
                                                                                nb_classes)
        store_models_accuracies.append(ensemble_model_accuracy)
        print("Fold %d : %f" % (test_index, ensemble_model_accuracy))
        ensemble_model_name = "Global_Ensemble_" + all_models_names_string + "_split_test" + str(test_index)
        store_models_predictions.append([ensemble_model_name, convert_array2listofarrays(ensemble_model_predictions)])

    # Save results in a csv file
    csv_file_path = os.path.join(results_folder, "global_ensemble_summed_prediction_results_" + str(
        nb_folds) + "_folds_" + all_models_names_string + "_.csv")
    predictions_results = pd.DataFrame(store_models_predictions, columns=["path", "predictions"])
    predictions_results.to_csv(csv_file_path)

    # Average accuracy of all the folds
    return np.mean(np.array(store_models_accuracies))


#########################################################################################
## Main program
#########################################################################################

def main(args):
    try:
        print(args.operation)
        if args.operation == "Confusion_matrices":
            if args.ensemble_type == "Unique":
                models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(args.folds_number,
                                                                                                args.trained_models_folder,
                                                                                                args.model_type,
                                                                                                args.training_condition,
                                                                                                args.classes_status,
                                                                                                args.optical_flow_status,
                                                                                                args.augmentation_status,
                                                                                                args.augmentation_frequency)
                test_predictions_file = lookFor_UniqueEnsemble_predictionsFile(args.folds_number,
                                                                               args.results_folder,
                                                                               args.model_type,
                                                                               args.training_condition,
                                                                               args.classes_status,
                                                                               args.optical_flow_status,
                                                                               args.augmentation_status,
                                                                               args.augmentation_frequency)

            else: #args.ensemble_type == "Global"
                # trained_models_subfolder path hard coded for the needs of an article but it needs to be made generic
                trained_models_subfolder = "Trained_models/5folds_TWOSTREAM_I3D_PRETRAINED_CS_unbalanced_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3"
                test_predictions_file = lookFor_GlobalEnsemble_predictionsFile(args.folds_number,
                                                                               args.results_folder,
                                                                               args.models_list)
            compute_confusion_matrices(trained_models_subfolder,
                                       test_predictions_file,
                                       args.ensemble_type,
                                       args.results_folder)



        elif args.operation == "Difference_matrices":
            if args.ensemble_type == "Unique":
                models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(args.folds_number,
                                                                                                args.trained_models_folder,
                                                                                                args.model_type,
                                                                                                args.training_condition,
                                                                                                args.classes_status,
                                                                                                args.optical_flow_status,
                                                                                                args.augmentation_status,
                                                                                                args.augmentation_frequency)
                test_predictions_file = lookFor_UniqueEnsemble_predictionsFile(args.folds_number,
                                                                               args.results_folder,
                                                                               args.model_type,
                                                                               args.training_condition,
                                                                               args.classes_status,
                                                                               args.optical_flow_status,
                                                                               args.augmentation_status,
                                                                               args.augmentation_frequency)
                ensemble_models_list = None
            else:
                # ensemble_type == "Global"
                trained_models_subfolder = "Trained_models/5folds_TWOSTREAM_I3D_PRETRAINED_CS_unbalanced_OF_FarneBack_onTheFly_AS_augmented_precomputed_Freq3"
                test_predictions_file = lookFor_GlobalEnsemble_predictionsFile(args.folds_number,
                                                                               args.results_folder,
                                                                               args.models_list)
                ensemble_models_list = args.models_list
            compute_difference_matrices(trained_models_subfolder,
                                        test_predictions_file,
                                        args.results_folder,
                                        args.ensemble_type,
                                        ensemble_models_list)
        elif args.operation == "Evaluate_ensembles":
            models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(args.folds_number,
                                                                                            args.trained_models_folder,
                                                                                            args.model_type,
                                                                                            args.training_condition,
                                                                                            args.classes_status,
                                                                                            args.optical_flow_status,
                                                                                            args.augmentation_status,
                                                                                            args.augmentation_frequency)
            test_probabilities_file = lookFor_probabilitiesFile(args.folds_number,
                                                                args.results_folder,
                                                                args.model_type,
                                                                args.training_condition,
                                                                args.classes_status,
                                                                args.optical_flow_status,
                                                                args.augmentation_status,
                                                                args.augmentation_frequency,
                                                                involved_sets="test")
            trainval_probabilities_file = lookFor_probabilitiesFile(args.folds_number,
                                                                    args.results_folder,
                                                                    args.model_type,
                                                                    args.training_condition,
                                                                    args.classes_status,
                                                                    args.optical_flow_status,
                                                                    args.augmentation_status,
                                                                    args.augmentation_frequency,
                                                                    involved_sets="train_val")

            evaluate_ensembles(trained_models_subfolder,
                               args.results_folder,
                               args.weights_type,
                               os.path.join(args.historiesFolder_validationErrorInverse,
                                            os.path.basename(trained_models_subfolder)),
                               test_probabilities_file,
                               trainval_probabilities_file,
                               args.weights_array_file,
                               args.batch_size,
                               args.workers,
                               args.model_type,
                               args.training_condition,
                               args.optical_flow_status,
                               args.augmentation_status,
                               args.augmentation_frequency,
                               args.classes_status,
                               models_name)
        elif args.operation == "Store_models_probabilities":
            models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(args.folds_number,
                                                                                            args.trained_models_folder,
                                                                                            args.model_type,
                                                                                            args.training_condition,
                                                                                            args.classes_status,
                                                                                            args.optical_flow_status,
                                                                                            args.augmentation_status,
                                                                                            args.augmentation_frequency)
            store_probabilities(trained_models_subfolder,
                                args.results_folder,
                                args.involved_sets,
                                args.batch_size,
                                args.workers,
                                args.model_type,
                                args.training_condition,
                                args.optical_flow_status,
                                args.augmentation_status,
                                args.augmentation_frequency,
                                args.classes_status,
                                models_name)
        elif args.operation == "StickDiagrams_wellClassifiedClips_per_numberOfModels":
            models_name, trained_models_subfolder = get_ModelsNameAndTrainedModelsSubfolder(args.folds_number,
                                                                                            args.trained_models_folder,
                                                                                            args.model_type,
                                                                                            args.training_condition,
                                                                                            args.classes_status,
                                                                                            args.optical_flow_status,
                                                                                            args.augmentation_status,
                                                                                            args.augmentation_frequency)
            test_probabilities_file = lookFor_probabilitiesFile(args.folds_number,
                                                                args.results_folder,
                                                                args.model_type,
                                                                args.training_condition,
                                                                args.classes_status,
                                                                args.optical_flow_status,
                                                                args.augmentation_status,
                                                                args.augmentation_frequency,
                                                                involved_sets="test")
            stickDiagrams_wellClassifiedClips_per_numberOfModels(trained_models_subfolder,
                                                                 test_probabilities_file,
                                                                 args.involved_sets,
                                                                 models_name,
                                                                 args.results_folder)
        elif args.operation == "Global_evaluate_models":
            print(args.models_list)
            print("Folds number : " + str(args.folds_number))
            print("Results folder : " + args.results_folder)
            print("Trained models folder : " + args.trained_models_folder)
            global_evaluate_ensembles(args.folds_number,
                                      args.trained_models_folder,
                                      args.models_list,
                                      args.results_folder)
        elif args.operation == "Combine_ensembles":
            print(args.models_list)
            print("Folds number : " + str(args.folds_number))
            print("Results folder : " + args.results_folder)
            print("Trained models folder : " + args.trained_models_folder)
            combine_ensembles(args.folds_number,
                              args.trained_models_folder,
                              args.models_list,
                              args.results_folder)
        else:
            print("Operation not mentioned")
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()


if __name__ == '__main__':
    ## ensure that the script is running on gpu (to apply locally on the Lab computer)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    ## clear session, in case it's necessary
    K.clear_session()

    ## verify that we are running on gpu
    if len(K.tensorflow_backend._get_available_gpus()) == 0:
        print('error-no-gpu')
        exit()
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-op',
                        '--operation',
                        help='Specify the operation to run.',
                        type=str,
                        choices=['Confusion_matrices', 'Difference_matrices', 'Evaluate_ensembles',
                                 'Store_models_probabilities', 'StickDiagrams_wellClassifiedClips_per_numberOfModels',
                                 'Global_evaluate_models', 'Combine_ensembles'],
                        required=True)

    parser.add_argument('-et',
                        '--ensemble_type',
                        help='Specify the type the ensemble on which to compute the confusion matrices.',
                        type=str,
                        choices=['Unique', 'Global'],
                        required=False)

    parser.add_argument('-mlist',
                        '--models_list',
                        nargs='+',
                        help='List the models that are involved in the global ensemble prediction. Example : python evaluate_ensemble.py -mlist TWOSTREAM_I3D_PRETRAINED C3D_SCRATCH',
                        required=False)

    parser.add_argument('-fn',
                        '--folds_number',
                        help='Specify the number of folds.',
                        type=int)

    parser.add_argument('-tmf',
                        '--trained_models_folder',
                        help='Specify the path to the pretrained_models folder.',
                        type=str)

    parser.add_argument('-rf',
                        '--results_folder',
                        help='Specify the path to the results folder.',
                        type=str,
                        default="Results",
                        required=True)

    parser.add_argument('-is',
                        '--involved_sets',
                        help="Coupled with Evaluate_models or Store_models_probabilities operations. It mentions which set to run the operation on.\
                        It is mandatory for Grid_Search and Differential_Evolution to compute weights on Train_val sets",
                        type=str,
                        default='test',
                        choices=['train_val', 'test'])

    parser.add_argument('-prf',
                        '--prediction_results_file',
                        help='Specify the path to the file that stores the prediction results.',
                        type=str)

    parser.add_argument('-wt',
                        '--weights_type',
                        help='Specify the weighting operation type to apply for the individual models.',
                        type=str,
                        choices=['GRID_SEARCH', 'DIFFERENTIAL_EVOLUTION', 'SUM', 'VALIDATION_ERROR_INVERSE', 'MAXIMUM'])

    parser.add_argument('-wf',
                        '--weights_array_file',
                        help='Path to the weights_array_file (numpy file).',
                        type=str)

    parser.add_argument('-hf_vei',
                        '--historiesFolder_validationErrorInverse',
                        help='Specify the path to the histories folder that is needed for the computation of the validation error inverse.',
                        default="Data/Weights",
                        type=str)

    parser.add_argument('-mt',
                        '--model_type',
                        help='Specify the model name.',
                        type=str,
                        choices=['TWOSTREAM_I3D', 'I3D', 'C3D', 'R3D_18', 'R3D_34', 'R3D_50', 'R3D_101', 'R3D_152'])

    parser.add_argument('-tc',
                        '--training_condition',
                        help='Specify how was the weights'' state of the model before going to be trained on Crowd-11.',
                        type=str,
                        choices=['_SCRATCH', '_PRETRAINED'])

    parser.add_argument('-as',
                        '--augmentation_status',
                        help='Mentions if we want to apply or not data augmentation.',
                        choices=['non_augmented', 'augmented_onTheFly', 'augmented_precomputed'],
                        type=str,
                        default='non_augmented')

    parser.add_argument('-af',
                        '--augmentation_frequency',
                        help='Associated with augmentation_status when it is set to augmented_precomputed.',
                        type=int,
                        default=0)

    parser.add_argument('-ofs',
                        '--optical_flow_status',
                        help='Specifies if the optical flow was pre-computed (happens with TV-L1) or is computed on-the-fly (happens with Farneback).',
                        type=str,
                        choices=['TVL1_precomputed', 'FarneBack_onTheFly'])

    parser.add_argument('-cs',
                        '--classes_status',
                        help='Mentions if we want to make the data balanced or keep it as is.',
                        choices=['balanced', 'unbalanced'],
                        type=str,
                        default='unbalanced')

    parser.add_argument('-w',
                        '--workers',
                        help='Specify the number of GPUs involved in the computation.',
                        type=int)

    parser.add_argument('-b',
                        '--batch_size',
                        help='Specify the batch_size for training.',
                        type=int)

    args = parser.parse_args()

    main(args)
