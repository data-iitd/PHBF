import pandas as pd
import data_loader
import time
import numpy as np

from compute import hpbf_selection_compute

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


def run_phbf(
    dname,
    dataset,
    low,
    high,
    num,
    epochs,
    hash_counts=None,
    hashes=None,
    sample_factor=100,
    bits_per_array=1000,
):
    x_train, y_train, x_test, y_test = dataset

    HASH_COUNTS = []
    SIZES = []
    TOTAL_SIZES = []
    TIMES = []
    FPRS = []
    STDS = []

    bitarray_sizes = np.linspace(high, low, num=num, dtype=np.int)

    logger.info(f"Running {dname} with PBF with Selection")
    for bitarray_size in bitarray_sizes:
        hash_count = int(bitarray_size / bits_per_array)
        result = hpbf_selection_compute(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            bitarray_size=bitarray_size,
            hash_count=hash_count,
            epochs=epochs,
            sample_factor=sample_factor,
        )

        HASH_COUNT, SIZE, TOTAL_SIZE, TIME, FPR, STD = result
        HASH_COUNTS.append(HASH_COUNT)
        SIZES.append(SIZE)
        TOTAL_SIZES.append(TOTAL_SIZE)
        TIMES.append(TIME)
        FPRS.append(FPR)
        STDS.append(STD)
        logger.info(
            f"Running: {hash_count} {bitarray_size} with PBF with Selection, FPR = {FPR}"
        )

    df = pd.DataFrame(
        {
            "HASH_COUNT": HASH_COUNTS,
            "BITARRAY_SIZE": SIZES,
            "TOTAL_SIZE": TOTAL_SIZES,
            "TIMES": TIMES,
            "FPRS": FPRS,
            "STDS": STDS,
        }
    )
    ctime = time.strftime("%d%H%M")
    filename = f"results/{dname}_{low}_{high}_{num}_hpbfS.csv"
    logger.info(f"Writing {filename}")
    df.to_csv(filename, index=False)


def run_covertype():
    dname = "COVERTYPE"
    logger.info(f"Loading {dname} with PBF")
    dataset = data_loader.load_covertype()
    low = 100000
    high = 10000000
    num = 10
    epochs = 10
    hash_counts = 3
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
    )


def run_kitsune():
    attack = "Mirai"
    dname = f"KITSUNE-{attack}"
    logger.info(f"Loading {dname}")
    dataset = data_loader.load_kitsune(attack=attack)
    low = 1000000
    high = 10000000
    num = 10
    epochs = 1
    hash_counts = 3
    # hashes = list(np.linspace(1, 30, num=5, dtype=int))
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
    )


def run_gaussian():
    dname = f"GAUSSIAN"
    logger.info(f"Loading {dname} with hpbfS")
    dataset = data_loader.load_gaussian(dimensions=(520, 3), separation=80)
    low = 10
    high = 50
    num = 10
    epochs = 1
    hash_counts = 1
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
    )

def run_mnist():
    dname = "MNIST"
    dataset = data_loader.load_mnist(pos_digits=[0], num_pos=6000, num_neg=6000)
    num = 10
    low = 6000
    high = 60000
    hash_counts = 3
    hashes = list(np.linspace(1, 30, num=5, dtype=int))
    epochs = 1
    sample_factor = 200

    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
        sample_factor=sample_factor,
        hashes=hashes,
    )


def run_facebook():
    dname = "FB_CHECKIN"
    logger.info(f"Loading {dname} with PBF")
    dataset = data_loader.load_facebook(augment_data=False)
    low = 50000
    high = 500000
    num = 10
    epochs = 1
    hash_counts = 3
    hashes = list(np.linspace(1, 30, num=5, dtype=int))
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
        hashes=hashes,
        sample_factor=200,
    )


def run_letor():
    dname = "LETOR"
    logger.info(f"Loading {dname} with PBF")
    dataset = data_loader.load_letor(augment_data=False)
    low = 1800
    high = 18000
    num = 10
    epochs = 1
    hash_counts = 3
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
    )


def run_miniboone():
    dname = "MINIBOONE"
    logger.info(f"Loading {dname} with PBF")
    dataset = data_loader.load_miniboone()
    low = 36500
    high = 365000
    num = 10
    epochs = 1
    hash_counts = 3
    hashes = [1, 10, 30]
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
        hashes=hashes,
    )


def run_higgs():
    dname = "HIGGS"
    logger.info(f"Loading {dname} with PBF")
    dataset = data_loader.load_higgs()
    low = 100
    high = 10000
    num = 10
    epochs = 3
    hash_counts = 3
    logger.info(f"Running {dname} with PBF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        hash_counts=hash_counts,
    )


def run_ember():
    dname = "EMBER"
    logger.info(f"Loading {dname} with BF")
    dataset = data_loader.load_ember()
    low = 10000
    high = 100000
    num = 5
    epochs = 3
    logger.info(f"Running {dname} with BF")
    run_phbf(
        dname=dname,
        dataset=dataset,
        low=low,
        high=high,
        num=num,
        epochs=epochs,
        bits_per_array=5000,
    )


if __name__ == "__main__":
    # run_covertype()
    # run_mnist()
    # run_facebook()
    # run_kitsune()
    # run_miniboone()
    run_gaussian()
    # run_letor()
    # run_higgs()
    # run_ember()
