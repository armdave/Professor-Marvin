import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

from parameters import *
from utils import OC_Statistics
from utils import getDataset, getDatasetNoMapTrain, getDatasetNoMapVal, plot_history
from get_data import downloadData, getDataDict, getDataframe


def create_train_and_val():
    """
    creates a training and validation set of type tf.data.dataset
    """

    # Download data
    downloadData(data_path="/input/speech_commands/")

    # Get data dictionary
    dataDict = getDataDict(data_path="/input/speech_commands/")

    # Obtain dataframe for each dataset
    trainDF = getDataframe(dataDict["train"])
    valDF = getDataframe(dataDict["val"])
    devDF = getDataframe(dataDict["dev"])
    testDF = getDataframe(dataDict["test"])

    print("Dataset statistics")
    print("Train files: {}".format(trainDF.shape[0]))
    print("Validation files: {}".format(valDF.shape[0]))
    print("Dev test files: {}".format(devDF.shape[0]))
    print("Test files: {}".format(testDF.shape[0]))

    # Use TF Data API for efficient data input
    train_data, train_steps = getDatasetNoMapTrain(df=trainDF, batch_size=BATCH_SIZE, cache_file="train_cache", shuffle=True)

    val_data, val_steps = getDatasetNoMapVal(df=valDF, batch_size=BATCH_SIZE, cache_file="val_cache", shuffle=False)
    return train_data, train_steps, val_data, val_steps

    
