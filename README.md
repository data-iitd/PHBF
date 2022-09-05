# Hash Partitioned Bloom Filter

This repository contains the code for VLDB 2022 paper: [New wine in an old bottle: data-aware hash functions for bloom filters](https://dl.acm.org/doi/abs/10.14778/3538598.3538613?casa_token=4AEbgykagvYAAAAA:7P0EX9EpjogccBKL4hbZPM04wLSO31tIR7iNKI5lGguZrRgg1LlitfubFiDIr0tBKqzhz8NSyPaZqQ)

## Usage

```python
from filters import HPBF
from data_loader import load_mnist

bitarray_size = 1000 # m
hash_count = 12 # k
input_dim = 784 # d
sample_factor = 8 # s

hpbf = HPBF(
            bitarray_size,
            hash_count,
            input_dim,
            sample_factor=sample_factor,
        )

x_train, _, x_test, _ = load_mnist() # The sets X and Y of paper

hpbf.initialize(x_train, x_test) # select the vectors
hpbf.bulk_add(x_train) # compute hashes and populate the filter
fpr = hpbf.compute_fpr(x_test)
```

## Reference

Please use to following reference when citing our work

```
Arindam Bhattacharya, Chathur Gudesa, Amitabha Bagchi, and Srikanta Bedathur. 2022. New wine in an old bottle: data-aware hash functions for bloom filters. Proc. VLDB Endow. 15, 9 (May 2022), 1924–1936. https://doi.org/10.14778/3538598.3538613
```

```
@article{10.14778/3538598.3538613,
author = {Bhattacharya, Arindam and Gudesa, Chathur and Bagchi, Amitabha and Bedathur, Srikanta},
title = {New Wine in an Old Bottle: Data-Aware Hash Functions for Bloom Filters},
year = {2022},
issue_date = {May 2022},
publisher = {VLDB Endowment},
volume = {15},
number = {9},
issn = {2150-8097},
url = {https://doi.org/10.14778/3538598.3538613},
doi = {10.14778/3538598.3538613},
abstract = {In many applications of Bloom filters, it is possible to exploit the patterns present in the inserted and non-inserted keys to achieve more compression than the standard Bloom filter. A new class of Bloom filters called Learned Bloom filters use machine learning models to exploit these patterns in the data. In practice, these methods and their variants raise many questions: the choice of machine learning models, the training paradigm to achieve the desired results, the choice of thresholds, the number of partitions in case multiple partitions are used, and other such design decisions. In this paper, we present a simple partitioned Bloom filter that works as follows: we partition the Bloom filter into segments, each of which uses a simple projection-based hash function computed using the data. We also provide a theoretical analysis that provides a principled way to select the design parameters of our method: number of hash functions and number of bits per partition. We perform empirical evaluations of our methods on various real-world datasets spanning several applications. We show that it can achieve an improvement in false positive rates of up to two orders of magnitude over standard Bloom filters for the same memory usage, and upto 50% better compression (bytes used per key) for same FPR, and, consistently beats the existing variants of learned Bloom filters.},
journal = {Proc. VLDB Endow.},
month = {may},
pages = {1924–1936},
numpages = {13}
}
```

## Disclaimer

This is released as a research prototype. It is not meant to be a production quality implementation. It has been made open source to enable easy reproducibility of research results.