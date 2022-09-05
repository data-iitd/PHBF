import numpy as np
import pandas as pd
import sklearn.datasets as skds
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# import ember

# from imblearn.over_sampling import SMOTE

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("logs/agreement_all.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ALL = 10000000000000


def load_mnist(pos_digits=[0], num_pos=6000, num_neg=6000):
    x, y = skds.fetch_openml("mnist_784", return_X_y=True)
    x = np.array(x)
    y = np.array(y, dtype=np.int)
    conditions = []
    for i in range(len(y)):
        if y[i] in pos_digits:
            conditions.append(True)
        else:
            conditions.append(False)
    conditions = np.array(conditions)
    x_train = x[conditions]
    index = np.random.choice(len(x_train), num_pos, replace=False)
    x_train = x_train[index]
    scalar = MinMaxScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    y_train = np.ones(len(x_train))

    x_test = x[~conditions]
    index = np.random.choice(len(x_test), num_neg, replace=False)
    x_test = x_test[index]
    x_test = scalar.fit_transform(x_test)
    y_test = np.zeros(len(x_test))

    logger.info(f"MNIST Train size = {x_train.shape[0]}")
    logger.info(f"MNIST Test size = {x_test.shape[0]}")
    logger.info(f"MNIST dimensions = {x_test.shape[1]}")
    return x_train, y_train, x_test, y_test


def load_covertype(pos_classes=[2], num_pos=ALL, num_neg=ALL):
    x, y = skds.fetch_openml(data_id=1596, return_X_y=True)
    x = np.array(x)
    y = np.array(y, dtype=np.int)

    conditions = []
    for i in range(len(y)):
        if y[i] in pos_classes:
            conditions.append(True)
        else:
            conditions.append(False)

    conditions = np.array(conditions)
    x_train = x[conditions]
    if len(x_train) > num_pos:
        index = np.random.choice(len(x_train), num_pos, replace=False)
        x_train = x_train[index]
    scalar = MinMaxScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    y_train = np.ones(len(x_train))

    x_test = x[~conditions]
    if len(x_test) > num_neg:
        index = np.random.choice(len(x_test), num_neg, replace=False)
        x_test = x_test[index]
    x_test = scalar.fit_transform(x_test)
    y_test = np.zeros(len(x_test))

    logger.info(f"Train size = {x_train.shape[0]}")
    logger.info(f"Test size = {x_test.shape[0]}")

    return x_train, y_train, x_test, y_test


def load_kitsune(attack="Mirai"):
    x = pd.read_csv(f"../data/kitsune/{attack}_dataset.csv").to_numpy()
    y = pd.read_csv(f"../data/kitsune/{attack}_labels.csv").to_numpy()

    y_tmp = []
    for val in y:
        y_tmp.append(val[0])
    y = np.array(y_tmp)

    x_train = x[y == 1]
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = np.ones(x_train.shape[0])

    x_test = x[y == 0]
    x_test = scaler.fit_transform(x_test)
    y_test = np.zeros(x_test.shape[0])

    logger.info(f"KITSUNE {attack} Train size = {x_train.shape[0]}")
    logger.info(f"KITSUNE {attack} Test size = {x_test.shape[0]}")
    logger.info(f"KITSUNE {attack} dimensions = {x_test.shape[1]}")

    return x_train, y_train, x_test, y_test


def load_gaussian(dimensions, separation, scale=1.0):
    mean = [separation]
    for i in range(dimensions[1] - 1):
        mean.append(0)

    x_train = np.random.normal(loc=0, scale=scale, size=dimensions)
    y_train = np.ones(x_train.shape[0])

    x_test = np.random.normal(loc=mean, scale=scale, size=dimensions)
    y_test = np.zeros(x_test.shape[0])

    # logger.info(f"Train size = {x_train.shape[0]}")
    # logger.info(f"Test size = {x_test.shape[0]}")
    # logger.info(f"Dimensions = {x_train.shape[1]}")
    
    return x_train, y_train, x_test, y_test


def load_magic():

    dataset = pd.read_csv("../datasets/magic04.data").to_numpy()

    x = []
    y = []
    for row in dataset:
        x.append(row[:-1])
        y.append(row[-1])

    x = np.array(x)
    y = np.array(y)

    condlist = [y == "g", y == "h"]
    choicelist = [1, 0]
    y = np.select(condlist, choicelist)

    scalar = MinMaxScaler().fit(x)
    X = scalar.transform(x)

    x_neg = X[y == 0]
    x_pos = X[y == 1]

    np.random.shuffle(x_pos)
    np.random.shuffle(x_neg)

    x_train = x_pos
    y_train = np.ones(len(x_pos))

    x_test = x_neg
    y_test = np.zeros(len(x_neg))

    logger.info(f"Train size = {x_train.shape[0]}")
    logger.info(f"Test size = {x_test.shape[0]}")
    return x_train, y_train, x_test, y_test


def load_miniboone():
    x = pd.read_csv("../datasets/MiniBooNE_PID.txt", delim_whitespace=True).to_numpy()
    y = np.concatenate((np.ones(36499), np.zeros(93564)))
    x_neg = x[y == 0]
    x_pos = x[y == 1]

    scaler = MinMaxScaler().fit(x_pos)
    x_pos = scaler.transform(x_pos)
    x_neg = scaler.transform(x_neg)

    np.random.shuffle(x_pos)
    np.random.shuffle(x_neg)

    x_train = x_pos
    y_train = np.ones(len(x_train))

    x_test = x_neg[: len(x_pos)]
    y_test = np.zeros(len(x_test))

    logger.info(f"MINIBOONE Train size = {x_train.shape[0]}")
    logger.info(f"MINIBOONE Test size = {x_test.shape[0]}")
    logger.info(f"MINIBOONE dimensions = {x_test.shape[1]}")

    return x_train, y_train, x_test, y_test


def load_facebook(augment_data=True):
    ds = pd.read_pickle("../data/facebook/facebook_checkin.pkl")
    x = ds.drop(["place_id"], axis=1)
    x = x.to_numpy()
    y = ds["place_id"].to_numpy()

    x_neg = x[y == 0]
    x_pos = x[y == 1]

    scaler = MinMaxScaler().fit(x_pos)
    x_pos = scaler.transform(x_pos)
    x_neg = scaler.transform(x_neg)

    np.random.shuffle(x_pos)
    np.random.shuffle(x_neg)

    x_train = x_pos
    y_train = np.ones(len(x_pos))

    x_test = x_neg
    y_test = np.zeros(len(x_neg))

    if augment_data:
        for i in range(4):
            new_x = x_train + np.random.uniform(low=-1, high=1, size=1)[0] * (1e-3)
            x_train = np.concatenate((x_train, new_x))

        for i in range(2):
            new_x = x_test + np.random.uniform(low=-1, high=1, size=1)[0] * (1e-3)
            x_test = np.concatenate((x_test, new_x))

    y_train = np.ones(x_train.shape[0])
    y_test = np.zeros(x_test.shape[0])

    logger.info(f"FACEBOOK train size = {x_train.shape[0]}")
    logger.info(f"FACEBOOK test size = {x_test.shape[0]}")
    logger.info(f"FACEBOOK dimensions = {x_train.shape[1]}")
    logger.info(f"Is augmented = {augment_data}")
    return x_train, y_train, x_test, y_test


def load_higgs():
    ds = pd.read_csv("../data/HIGGS/HIGGS.csv")
    x = ds.drop(["label"], axis=1)
    x = x.to_numpy()
    y = ds["label"].to_numpy()

    x_neg = x[y == 0]
    x_pos = x[y == 1]

    np.random.shuffle(x_pos)
    np.random.shuffle(x_neg)

    scaler = MinMaxScaler().fit(x_pos)
    x_pos = scaler.transform(x_pos)
    x_neg = scaler.transform(x_neg)

    x_train = x_pos
    y_train = np.ones(len(x_pos))

    x_test = x_neg
    y_test = np.zeros(len(x_neg))

    logger.info(f"HIGGS train size = {x_train.shape[0]}")
    logger.info(f"HIGGS test size = {x_test.shape[0]}")
    logger.info(f"HIGGS dimensions = {x_train.shape[1]}")
    return x_train, y_train, x_test, y_test


def load_ember():
    x_train, y_train, x_test, y_test = ember.read_vectorized_features(
        "../data/EMBER/ember2018/"
    )
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x_neg = x[y == 0]
    x_pos = x[y == 1]

    scaler = MinMaxScaler().fit(x_pos)
    x_pos = scaler.transform(x_pos)
    x_neg = scaler.transform(x_neg)

    np.random.shuffle(x_pos)
    np.random.shuffle(x_neg)

    x_train = x_pos
    y_train = np.ones(len(x_pos))

    x_test = x_neg
    y_test = np.zeros(len(x_neg))

    logger.info(f"EMBER train size = {x_train.shape[0]}")
    logger.info(f"EMBER test size = {x_test.shape[0]}")
    logger.info(f"EMBER dimensions = {x_train.shape[1]}")
    return x_train, y_train, x_test, y_test