# ECC datasets

These datasets were captured from the NXP K82F dedicated cryptographic
accelerator (LTC -- LP Trusted Crypto) as a base for all implementations to
perform constant-time hardware-accelerated scalar multiplication and point
addition. Countermeasures have been done in software using hardware accelerated
scalar multiplication and point addition as primitives. For more details see
[Generalized Power Attacks against Crypto Hardware using Long-Range Deep
Learning](https://github.com/google/scaaml/tree/main/papers/2024/GPAM).

## Where to download

[Install gsutil](https://cloud.google.com/storage/docs/gsutil_install).

```bash
# List available datasets
gcloud storage ls gs://scaaml-public/datasets/ECC/GPAM
# Show size of the directory
gcloud storage du --summarize --readable-sizes gs://scaaml-public/datasets/ECC/GPAM

# Download all (beware that the multiprocessing option -m can saturate your
# network connection). Or download just the dataset you want.
gcloud storage rsync --recursive gs://scaaml-public/datasets/ECC/GPAM .
```

## How to load and use a dataset

```python
from scaaml.io import Dataset
from scaaml.stats import ExampleIterator


ds_path = "K82F_ECC_CM0_ECC-FR256_CW308"

# Print a summary.
Dataset.summary(ds_path)

# Load the dataset.
dataset = Dataset.from_config(ds_path)
# Check consistency of download (control hash-sums).
dataset.check(key_ap="k")

# Iterate examples one by one.
# The argument `split` can be set to:
#   Dataset.TRAIN_SPLIT   training data
#   Dataset.TEST_SPLIT    validation data
#   Dataset.HOLDOUT_SPLIT holdout data
for example in ExampleIterator(dataset_path=ds_path, split=Dataset.TEST_SPLIT):
  print(example)

# Or use `dataset.as_tfdataset` to iterate batches using `tf.data.Dataset`.
```

## ECC CM0

Single hardware accelerated scalar multiplication.

## ECC CM1

Additive blinding using a 256-bit long random mask.

## ECC CM2

Multiplicative blinding using a 128-bit long random mask.

## ECC CM3

Combination of CM1 and CM2 (CM1 where each scalar multiplication in CM1 is
protected by an independent CM2).

## ECC Firmware

A firmware equivalent to the one used for the GPAM paper has been merged to the
ChipWhisperer repository
[simpleserial-ecc.c](https://github.com/newaetech/chipwhisperer/blob/develop/firmware/mcu/simpleserial-ecc-notrace/simpleserial-ecc.c)
([commit
6f0b3e0b0bc3f8423c6ee0428ae9712ad5412454](https://github.com/newaetech/chipwhisperer/commit/6f0b3e0b0bc3f8423c6ee0428ae9712ad5412454)).
Note that we have not tested 1:1 compatibility with the pre-trained models.
Notable differences:

-   Use of the ChipWhisperer firmware stack vs bare-metal dedicated firmware
    used in the paper. Impact should be negligible when the trigger is high but
    there could be timing differences when implementing a countermeasure (e.g.,
    CM1 with two multiplications and one addition orchestrated by a Python
    code).
-   K82F target chip clocked at 7.37MHz is the same as in the paper.
-   Serial communication is at 38400 bauds with ChipWhisperer whereas our paper
    had 115200 bauds.
