import math
import time
import numpy as np

from filters import (
    HPBF,
    ProjectionBloomFilter,
    ProjectionBloomFilterWithSelection,
    MultipleBitarrayProjectionBloomFilter,
    KraskaBloomFilter,
    BloomFilter,
)

from LBF import BloomClassifier

from bitarray import bitarray

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


def bf_compute(x_train, y_train, x_test, y_test, bitarray_size, hash_count):
    bf = BloomFilter(bitarray_size, hash_count)
    TIME = 0.0
    logger.info(f"BF adding points into filter")
    start = time.time()
    bf.bulk_add(x_train)
    end = time.time()
    TIME = end - start
    logger.info(f"BF computing FPR")
    fpr = bf.compute_fpr(x_test)
    logger.info(f"PBF number of buckets filled: {bf.filled()}")
    total_size = bitarray_size
    return hash_count, bitarray_size, total_size, TIME, fpr


def pbf_compute(x_train, y_train, x_test, y_test, bitarray_size, hash_count, epochs):

    input_dim = x_train.shape[1]
    TIME = 0.0
    fprs = []

    for epoch in range(epochs):
        logger.info(f"PBF computing epoch {epoch}")
        pbf = ProjectionBloomFilter(bitarray_size, hash_count, input_dim)
        start = time.time()
        pbf.bulk_add(x_train)
        end = time.time()
        TIME += end - start
        fpr = pbf.compute_fpr(x_test)
        fprs.append(fpr)
    logger.info(f"PBF number of buckets filled: {pbf.filled()}")
    total_size = x_train.shape[1] * hash_count * 16 + bitarray_size
    fpr, std = np.mean(fprs), np.std(fprs)
    return hash_count, bitarray_size, total_size, TIME, fpr, std


def pbf_selection_compute(
    x_train, y_train, x_test, y_test, bitarray_size, hash_count, epochs
):

    input_dim = x_train.shape[1]
    TIME = 0.0
    fprs = []

    for epoch in range(epochs):
        logger.info(f"PBF with selection: computing epoch {epoch}")
        pbfs = ProjectionBloomFilterWithSelection(
            bitarray_size, hash_count, input_dim, sample_factor=100, method="uniform"
        )
        start = time.time()
        pbfs.initialize(x_train, x_test)  # corresponding to X, Y of paper
        pbfs.bulk_add(x_train)
        end = time.time()
        TIME += end - start
        fpr = pbfs.compute_fpr(x_test)
        fprs.append(fpr)
    total_size = x_train.shape[1] * hash_count * 16 + bitarray_size
    fpr, std = np.mean(fprs), np.std(fprs)
    return hash_count, bitarray_size, total_size, TIME, fpr, std


def mbpbf_compute(x_train, y_train, x_test, y_test, bitarray_size, hash_count, epochs):

    input_dim = x_train.shape[1]
    TIME = 0.0
    fprs = []

    for epoch in range(epochs):
        logger.info(f"MBPBF computing epoch {epoch}")
        mbpbf = MultipleBitarrayProjectionBloomFilter(
            bitarray_size, hash_count, input_dim
        )
        start = time.time()
        mbpbf.bulk_add(x_train)
        end = time.time()
        TIME += end - start
        fpr = mbpbf.compute_fpr(x_test)
        fprs.append(fpr)

    total_size = x_train.shape[1] * hash_count * 16 + bitarray_size
    fpr, std = np.mean(fprs), np.std(fprs)
    return hash_count, bitarray_size, total_size, TIME, fpr, std

# optimistic
# gaussian

def hpbf_selection_compute(
    x_train,
    y_train,
    x_test,
    y_test,
    bitarray_size,
    hash_count,
    epochs,
    sample_factor,
    method="optimistic",
):

    input_dim = x_train.shape[1]
    TIME = 0.0
    fprs = []

    for epoch in range(epochs):
        # logger.info(f"hpbf with selection: computing epoch {epoch}")
        pbfs = HPBF(
            bitarray_size,
            hash_count,
            input_dim,
            sample_factor=sample_factor,
            method=method
        )
        start = time.time()
        pbfs.initialize(x_train, x_test)  # corresponding to X, Y of paper
        pbfs.bulk_add(x_train)
        end = time.time()
        TIME += end - start
        fpr = pbfs.compute_fpr(x_test)
        fprs.append(fpr)
    total_size = x_train.shape[1] * hash_count * 16 + bitarray_size
    fpr, std = np.mean(fprs), np.std(fprs)
    return hash_count, bitarray_size, total_size, TIME, fpr, std


def lbf_compute(x_train, y_train, x_test, y_test, bitarray_size, hash_count):

    TIME = 0.0
    lbf = BloomClassifier()
    init_x = np.concatenate((x_train, x_test))
    init_y = np.concatenate((y_train, y_test))
    start = time.time()
    logger.info(f"LBF training model")
    lbf.initialize(init_x, init_y)
    logger.info(f"LBF insering data")
    lbf.insert(x_train, m=bitarray_size, k=hash_count)
    end = time.time()
    TIME = end - start
    logger.info(f"LBF computing fpr")
    fpr = lbf.compute_fpr(x_test)
    total_size = (
        bitarray_size + sum([a.shape[0] * a.shape[1] for a in lbf.clf.coefs_]) * 32
    )
    return hash_count, bitarray_size, total_size, TIME, fpr


def kbf_compute(model, train_predictions, test_predictions, bitarray_size):

    kbf = KraskaBloomFilter(model)
    fpr_values = []

    kbf.create_filter(bitarray_size)

    logger.info(f"KBF computing insertion positions")
    for prediction in train_predictions:
        result = math.floor(prediction * (bitarray_size - 1))
        kbf.bit_array[result] = 1
    logger.info(f"KBF computed insertion positions")
    logger.info(f"KBF number of buckets filled: {kbf.filled()}")

    fp = 0
    tn = 0
    logger.info(f"KBF computing query positions")
    for prediction in test_predictions:
        result = math.floor(prediction * (bitarray_size - 1))
        if kbf.bit_array[result] == 1:
            fp += 1
        else:
            tn += 1
    logger.info(f"KBF computed query positions")
    fpr = fp / (fp + tn)
    return fpr