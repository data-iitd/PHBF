import numpy as np
import math
from bitarray import bitarray
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
import mmh3

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


class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, x):
        for seed in range(0, self.hash_count):
            result = mmh3.hash(bytes(x), seed) % self.size
            self.bit_array[result] = 1

    def lookup(self, x):
        for seed in range(0, self.hash_count):
            result = mmh3.hash(bytes(x), seed) % self.size
            if self.bit_array[result] == 0:
                return False
        return True

    def bulk_add(self, X):
        for x in X:
            self.add(x)

    def compute_fpr(self, X):
        fp = 0
        tn = 0
        for x in X:
            if self.lookup(x):
                fp += 1
            else:
                tn += 1
        return fp / (fp + tn)

    def get_efficient_hash_count(bitarray_size, num_inserted):
        eff_hash_count = int(math.log(2) * bitarray_size / num_inserted)
        eff_hash_count = max(eff_hash_count, 1)
        return eff_hash_count

    def compute_fpr_theoretical(bitarray_size, num_inserted):
        m = bitarray_size
        n = num_inserted
        k = BloomFilter.get_efficient_hash_count(m, n)
        return (1 - (1 - (1 / m)) ** (n * k)) ** k

    def compute_fprs_theoretical(bitarray_sizes, num_inserted):
        n = num_inserted
        fprs = []
        for m in bitarray_sizes:
            k = BloomFilter.get_efficient_hash_count(m, n)
            fpr = (1 - (1 - (1 / m)) ** (n * k)) ** k
            fprs.append(fpr)
        return fprs

    def filled(self):
        cnt = 0
        for i in range(self.size):
            if self.bit_array[i] == 1:
                cnt += 1
        return cnt


class ProjectionBloomFilter:
    def __init__(self, size, hash_count, dim):
        # vectors = np.random.normal(
        #     0, 1, size=(hash_count, dim)
        # )  # Choose random vectors from a Gaussian with mean 0 and variance 1
        vectors = np.random.uniform(0, 1, size=(hash_count, dim))
        self._normalize_vectors(vectors)
        self.vectors = np.transpose(vectors)  # Each column corresponds to a vector
        self.hash_count = hash_count
        self.size = size
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _normalize_vectors(
        self, vectors
    ):  # Converts an array where each row is a vector to corresponding unit vectors
        for i in range(len(vectors)):
            vectors[i] = vectors[i] / np.sqrt(np.sum(vectors[i] ** 2))
        return vectors

    def compute_hashes(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        projections = np.transpose(projections)
        # projections = expit(projections)  # Sigmoid on each value so that they are in the range (0,1)

        # All values are integers in the range [0, bitarray_size-1]
        hash_values = (projections * (self.size - 1)).astype(int)
        return hash_values  # Each row contains hash values of that datapoint corresponding to that row

    def bulk_add(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        self.scaler = [
            MinMaxScaler().fit(projections[i].reshape(-1, 1))
            for i in range(projections.shape[0])
        ]
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        # projections = expit(projections)
        projections = np.transpose(projections)
        hash_values = (projections * (self.size - 1)).astype(int)

        for row in hash_values:
            for hash_value in row:
                self.bit_array[hash_value] = 1

    def lookup(self, X):
        results = np.full(X.shape[0], True)
        hash_values = self.compute_hashes(X)
        for i in range(hash_values.shape[0]):
            for hash_value in hash_values[i]:
                if (
                    hash_value < 0
                    or hash_value >= self.size
                    or self.bit_array[hash_value] == 0
                ):
                    results[i] = False
                    break
        return results

    def compute_fpr(self, X):  # Assumes that X contains only negative samples
        fp = 0
        tn = 0
        results = self.lookup(X)
        for result in results:
            if result == True:
                fp += 1
            else:
                tn += 1
        return fp / (fp + tn)

    def filled(self):
        cnt = 0
        for i in range(self.size):
            if self.bit_array[i] == 1:
                cnt += 1
        return cnt


class MultipleBitarrayProjectionBloomFilter:
    def __init__(self, size, hash_count, dim):
        vectors = np.random.normal(
            0, 1, size=(hash_count, dim)
        )  # Choose random vectors from a Gaussian with mean 0 and variance 1
        self._normalize_vectors(vectors)
        self.vectors = np.transpose(vectors)  # Each column corresponds to a vector
        self.hash_count = hash_count
        self.size = int(size / hash_count)
        self.bit_array = [bitarray(self.size) for i in range(hash_count)]
        for i in range(len(self.bit_array)):
            self.bit_array[i].setall(0)

    def _normalize_vectors(
        self, vectors
    ):  # Converts an array where each row is a vector to corresponding unit vectors
        for i in range(len(vectors)):
            vectors[i] = vectors[i] / np.sqrt(np.sum(vectors[i] ** 2))
        return vectors

    def compute_hashes(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = expit(
            projections
        )  # Sigmoid on each value so that they are in the range (0,1)
        hash_values = (projections * (self.size - 1)).astype(
            int
        )  # All values are integers in the range [0, bitarray_size-1]
        return hash_values  # Each row contains hash values of that datapoint corresponding to that row

    def bulk_add(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        self.scaler = [
            MinMaxScaler().fit(projections[i].reshape(-1, 1))
            for i in range(projections.shape[0])
        ]
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        # projections = expit(projections)
        projections = np.transpose(projections)
        hash_values = (projections * (self.size - 1)).astype(int)

        for row in hash_values:
            for i in range(len(row)):
                hash_value = row[i]
                self.bit_array[i][hash_value] = 1

    def lookup(self, X):
        results = np.full(X.shape[0], True)
        hash_values = self.compute_hashes(X)
        for i in range(hash_values.shape[0]):
            for j in range(len(hash_values[i])):
                hash_value = hash_values[i][j]
                if self.bit_array[j][hash_value] == 0:
                    results[i] = False
                    break
        return results

    def compute_fpr(self, X):  # Assumes that X contains only negative samples
        fp = 0
        tn = 0
        results = self.lookup(X)
        for result in results:
            if result == True:
                fp += 1
            else:
                tn += 1
        return fp / (fp + tn)


class HPBF:
    def __init__(self, size, hash_count, dim, sample_factor=100, method="gaussian"):
        self.hash_count = hash_count
        self.size = int(size / hash_count)
        self.bit_array = [bitarray(self.size) for i in range(hash_count)]
        for i in range(len(self.bit_array)):
            self.bit_array[i].setall(0)
        self.method = method
        self.sample_factor = sample_factor
        self.dim = dim

    def _select_vectors(self, X, Y):
        size = (self.hash_count * self.sample_factor, self.dim)
        if self.method == "gaussian":
            candidates = np.random.normal(size=size)
        elif self.method == "optimistic":
            candidates = []
            for i in range(size[0]):
                vector = [1] # WARNING: WORST VECTORS
                for j in range(size[1] - 1):
                    vector.append(0)
                candidates.append(vector)
            candidates = np.array(candidates)

        else:
            candidates = np.random.uniform(size=size)

        pos_projections = np.dot(X, candidates.transpose())
        pos_projections = np.transpose(pos_projections)

        neg_projections = np.dot(Y, candidates.transpose())
        neg_projections = np.transpose(neg_projections)

        scaler = [
            MinMaxScaler().fit(pos_projections[i].reshape(-1, 1))
            for i in range(pos_projections.shape[0])
        ]
        for i in range(pos_projections.shape[0]):
            pos_projections[i] = (
                scaler[i].transform(pos_projections[i].reshape(-1, 1)).reshape(1, -1)[0]
            )
        for i in range(neg_projections.shape[0]):
            neg_projections[i] = (
                scaler[i].transform(neg_projections[i].reshape(-1, 1)).reshape(1, -1)[0]
            )

        pos_hash_values = (pos_projections * (self.size - 1)).astype(int)
        neg_hash_values = (neg_projections * (self.size - 1)).astype(int)
        assert pos_hash_values.shape[0] == candidates.shape[0]
        overlaps = []
        for pos_array, neg_array in zip(pos_hash_values, neg_hash_values):
            overlaps.append(len(np.intersect1d(pos_array, neg_array)))

        assert len(overlaps) == candidates.shape[0]
        overlaps = np.array(overlaps)
        best_hashes_idx = np.argsort(overlaps)
        best_hashes = candidates[best_hashes_idx[: self.hash_count]]
        return best_hashes

    def _normalize_vectors(
        self, vectors
    ):  # Converts an array where each row is a vector to corresponding unit vectors
        for i in range(len(vectors)):
            vectors[i] = vectors[i] / np.sqrt(np.sum(vectors[i] ** 2))
        return vectors

    def initialize(self, X, Y):
        """Initialize PBF

        Parameters
        ----------
        X : np.array
            Data to be inserted
        Y : np.array
            Data to be queried on
        """
        vectors = self._select_vectors(X, Y)
        self._normalize_vectors(vectors)
        self.vectors = np.transpose(vectors)  # Each column corresponds to a vector

    def compute_hashes(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        # print(projections[0][:3])
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        # print(projections[0][:3])
        projections = np.transpose(projections)
        # projections = expit(projections)  # Sigmoid on each value so that they are in the range (0,1)

        # All values are integers in the range [0, bitarray_size-1]
        for i in range(projections.shape[0]):
            for j in range(projections.shape[1]):
                if projections[i][j] > 1 or projections[i][j] < 0:
                    projections[i][j] = -1
                else:
                    projections[i][j] = (projections[i][j] * (self.size - 1))
        hash_values = projections.astype(int)
        # print(hash_values)
        # hash_values = (projections * (self.size - 1)).astype(int)
        return hash_values  # Each row contains hash values of that datapoint corresponding to that row

    def bulk_add(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        self.scaler = [
            MinMaxScaler().fit(projections[i].reshape(-1, 1))
            for i in range(projections.shape[0])
        ]
        # print(self.scaler[0].data_max_)
        # print(self.scaler[0].data_min_)
        # print(projections[0][:3])
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        # print(projections[0][:3])
        # projections = expit(projections)
        projections = np.transpose(projections)
        hash_values = (projections * (self.size - 1)).astype(int)

        for row in hash_values:
            for i in range(len(row)):
                hash_value = row[i]
                self.bit_array[i][hash_value] = 1

    def lookup(self, X):
        results = np.full(X.shape[0], True)
        hash_values = self.compute_hashes(X)
        for i in range(hash_values.shape[0]):
            for j in range(len(hash_values[i])):
                hash_value = hash_values[i][j]
                if hash_value != -1:
                    if self.bit_array[j][hash_value] == 0:
                        results[i] = False
                        break
                else:
                    # print("ERROR", hash_value)
                    results[i] = False
                    break
        return results

    def compute_fpr(self, X):  # Assumes that X contains only negative samples
        fp = 0
        tn = 0
        results = self.lookup(X)
        for result in results:
            if result == True:
                fp += 1
            else:
                tn += 1
        return fp / (fp + tn)

class ProjectionBloomFilterWithSelection:
    def __init__(self, size, hash_count, dim, sample_factor=100, method="gaussian"):
        """
        Parameters
        ----------
        size : int
            Size of the bit array
        hash_count : int
            Number of random vectors to use as hash functions
        dim : int
            Dimensionality of data
        sample_factor : int, optional
            sample_factor * hash_count vectors are sampled while selecting vectors, by default 100
        method: string, "gaussian" or "uniform"
            the distribution of sampled vectors
        """
        self.hash_count = hash_count
        self.size = size
        self.dim = dim
        self.sample_factor = sample_factor
        self.method = method
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _select_vectors(self, X, Y):
        size = (self.hash_count * self.sample_factor, self.dim)
        if self.method == "gaussian":
            candidates = np.random.normal(size=size)
        else:
            candidates = np.random.uniform(size=size)

        pos_projections = np.dot(X, candidates.transpose())
        pos_projections = np.transpose(pos_projections)

        neg_projections = np.dot(Y, candidates.transpose())
        neg_projections = np.transpose(neg_projections)

        scaler = [
            MinMaxScaler().fit(pos_projections[i].reshape(-1, 1))
            for i in range(pos_projections.shape[0])
        ]
        for i in range(pos_projections.shape[0]):
            pos_projections[i] = (
                scaler[i].transform(pos_projections[i].reshape(-1, 1)).reshape(1, -1)[0]
            )
        for i in range(neg_projections.shape[0]):
            neg_projections[i] = (
                scaler[i].transform(neg_projections[i].reshape(-1, 1)).reshape(1, -1)[0]
            )

        pos_hash_values = (pos_projections * (self.size - 1)).astype(int)
        neg_hash_values = (neg_projections * (self.size - 1)).astype(int)
        assert pos_hash_values.shape[0] == candidates.shape[0]
        overlaps = []
        for pos_array, neg_array in zip(pos_hash_values, neg_hash_values):
            overlaps.append(len(np.intersect1d(pos_array, neg_array)))

        assert len(overlaps) == candidates.shape[0]
        overlaps = np.array(overlaps)

        best_hashes_idx = np.argsort(overlaps)
        best_hashes = candidates[best_hashes_idx[: self.hash_count]]
        return best_hashes

        # for row in pos_hash_values:
        #     for hash_value in row:
        #         pos_bit_array[hash_value] = 1

        # for row in neg_hash_values:
        #     for hash_value in row:
        #         try:
        #             neg_bit_array[hash_value] = 1
        #         except IndexError as e:
        #             # print(e)
        #             pass

        # print(pos_bit_array)
        # print(neg_bit_array)

    def _normalize_vectors(self, vectors):
        """Converts an array where each row is a vector to corresponding unit vectors"""
        for i in range(len(vectors)):
            vectors[i] = vectors[i] / np.sqrt(np.sum(vectors[i] ** 2))
        return vectors

    def compute_hashes(self, X):
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        projections = np.transpose(projections)
        # projections = expit(projections)  # Sigmoid on each value so that they are in the range (0,1)

        # All values are integers in the range [0, bitarray_size-1]
        hash_values = (projections * (self.size - 1)).astype(int)
        return hash_values  # Each row contains hash values of that datapoint corresponding to that row

    def initialize(self, X, Y):
        """Initialize PBF

        Parameters
        ----------
        X : np.array
            Data to be inserted
        Y : np.array
            Data to be queried on
        """
        vectors = self._select_vectors(X, Y)
        self._normalize_vectors(vectors)
        self.vectors = np.transpose(vectors)  # Each column corresponds to a vector

    def bulk_add(self, X):
        """Inserts X into Bloom filter. Initialize first."""
        projections = np.dot(
            X, self.vectors
        )  # Projections of datapoints on the vectors
        projections = np.transpose(projections)
        self.scaler = [
            MinMaxScaler().fit(projections[i].reshape(-1, 1))
            for i in range(projections.shape[0])
        ]
        for i in range(projections.shape[0]):
            projections[i] = (
                self.scaler[i]
                .transform(projections[i].reshape(-1, 1))
                .reshape(1, -1)[0]
            )
        # projections = expit(projections)
        projections = np.transpose(projections)
        hash_values = (projections * (self.size - 1)).astype(int)

        for row in hash_values:
            for hash_value in row:
                self.bit_array[hash_value] = 1

    def lookup(self, X):
        results = np.full(X.shape[0], True)
        hash_values = self.compute_hashes(X)
        for i in range(hash_values.shape[0]):
            for hash_value in hash_values[i]:
                if (
                    hash_value < 0
                    or hash_value >= self.size
                    or self.bit_array[hash_value] == 0
                ):
                    results[i] = False
                    break
        return results

    def compute_fpr(self, X):  # Assumes that X contains only negative samples
        fp = 0
        tn = 0
        results = self.lookup(X)
        for result in results:
            if result == True:
                fp += 1
            else:
                tn += 1
        return fp / (fp + tn)

    def filled(self):
        cnt = 0
        for i in range(self.size):
            if self.bit_array[i] == 1:
                cnt += 1
        return cnt


class KraskaBloomFilter:
    def __init__(self, model):
        self.num_inserted = 0  # Keeps track of the number of elements that were inserted into the filter
        self.model = model  # classifier model associated as hash function
        self.size = None
        self.bit_array = None

    def create_filter(self, size):
        self.size = size
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def filled(self):
        cnt = 0
        for i in range(self.size):
            if self.bit_array[i] == 1:
                cnt += 1
        return cnt