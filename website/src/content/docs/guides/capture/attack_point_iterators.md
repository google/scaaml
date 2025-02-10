---
title: Attack Point Iterators
description: Attack Point Iterators
---

Training a deep learning network when the classes are not balanced is often
problematic.  In the situation when our dataset has 90% of cat images and only
10% of dog images the neural network may (and too often will) learn the
statistics rather than anything more useful.  In our toy example a network
which would classify each image as a "cat" would have 90% accuracy but would
definitely not be a reason for celebration.  If the classes were balanced
(roughly 50% of cat and roughly 50% of dog images) the risk of seeing this
would be much lower.

During profiling attacks we have a full control over the algorithm inputs.
SCAAML provides support for generators inspired by [latin
squares](https://en.wikipedia.org/wiki/Latin_square).  In combinatorics a latin
square is an `n` by `n` array filled with `n` different symbols, each occurring
exactly once in each column and exactly once in each row.  An example of a
latin square where `n = 3` is the following:
$$
\begin{matrix}
    1 & 2 & 3 \\
    3 & 1 & 2 \\
    2 & 3 & 1 \\
\end{matrix}
$$

In the case of AES128 we work with states of 16 bytes.  Let us imagine that we
want to generate balanced plaintexts.  Each byte of a plaintext can have 256
different values.  Our goal is generating 256 different plaintexts such that
each plaintext has uniformly and independently selected byte values.  That is
for each plaintext we have:

-   Each byte value is chosen uniformly at random (probability that is has
    value `x` is always 1/256).
-   A single byte value is
    [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory))
of other byte values.

The trick is to generate several (concretely a multiple of 256) plaintexts such
that values in one plaintext are not independent of the same byte index of
other plaintexts.  In the easiest case this would result in 256 plaintexts
which we could write as a table of 16 columns (one for each byte index) and 256
rows (one for each plaintext).  The goal is to have each column containing each
of the 256 values (0 to 255) exactly once.  Unlike latin squares we have no
condition on repetitions in rows (a plaintext can consist of 16 bytes each
equal to zero, even if the probability of this happening is very low).

To randomly generate such bunches of plaintexts we create a table of 16 columns
and 256 rows.  The row `i` consisting only of the number `i` (counted from zero
-- values 0 to 255).  Then we uniformly and independently shuffle each column
(using for instance the [Fisher-Yates
shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)).

A concrete example if we wanted to generate bits of a byte in such a way (just
two possible symbols for each of 8 positions).  We would create one row of 8
zeros and another row of 8 ones.  And one by one we would toss a random coin
for each of 8 columns either swapping the bit values or not.  This would result
in the first row representing a uniformly at random chosen byte.  The second
row would be the bitwise negation of the first line.

## Generators

As a convenience method the `scaaml.capture.input_generators` contains
`balanced_generator` which gives us balanced generation.

```python
from scaaml.capture.input_generators import balanced_generator

for value in balanced_generator(length=5, bunches=2, elements=3):
    # value is type npt.NDArray[np.int64]
    print(value)
```

A possible output is the following:

```bash
[0 1 2 2 1]
[1 2 1 0 2]
[2 0 0 1 0]
[1 1 0 1 1]
[2 0 2 2 2]
[0 2 1 0 0]
```

Notice that each row (each `value`) has length 5.  There are 3 possible
elements represented by values in `0..elements-1` (in Python language in
`range(elements)`).  We generated two bunches (first three and the last three)
of values.  In each bunch and each index of the value the elements are
different (e.g., first three lines if we look at the second column we see each
of the values 0, 1, and 2 once).

For completeness there is also `unrestricted_generator` generating uniformly
and independently at random with a similar API.

## Configuration-based Iterators

The usage of (balanced) generators depends on the application.  For training
purposes we might want to have balanced inputs of the SBOX.  To achieve this
for each key we could generate a single bunch or several bunches of plaintexts.
Generating balanced set of keys and for each of those a balanced set of
plaintexts would give us `256 * 256 = 65_536` inputs to capture with.

For a holdout split (a simulation of the attack) we would most likely want to
use a single key with several plaintexts each chosen independently at random.
This is to simulate a real world scenario when the byte values are not
balanced.

To support these two scenarios we could have the following code:

```python
# Capture train split:
for key in balanced_generator(length=16, bunches=1, elements=256):
  # We could also use the default values of bunches=1 and elements=256
  for plaintext in balanced_generator(length=16):
    # ChipWhisperer takes bytearray as input.
    key = bytearray(key.astype(np.uint8))
    plaintext = bytearray(plaintext.astype(np.uint8))
    print(f"TODO capture with {key = } {plaintext = }")

# Capture holdout split:
key = np.random.randint(low=0, high=256, size=16, dtype=np.int64)
for plaintext in unrestricted_generator(length=16, bunches=100):
  # ChipWhisperer takes bytearray as input.
  key = bytearray(key.astype(np.uint8))
  plaintext = bytearray(plaintext.astype(np.uint8))
  print(f"TODO capture with {key = } {plaintext = }")
```

If you decide to run the previous code it would print `256 * 256 + 256 * 100`
lines.

The situation gets more complicated when we need to generate also random masks
or move to another algorithms.  First of all generating balanced values using
three values would iterate `256 * 256 * 256 = 16_777_216` iterations.  Having
two random masks would create an unfeasibly large dataset.  In such a case we
might be tempted to generate several different possibilities each resulting in
a modification of the capture scripts (also different across the splits).

Luckily one can use the configuration based input generators provided by
`scaaml.capture.attack_point_iterators.attack_point_iterator`.  These provide
[itertools](https://docs.python.org/3/library/itertools.html) like building
blocks to ease the use.  These can be combined using human readable
[JSON](https://en.wikipedia.org/wiki/JSON) configurations.

-   constants: hardcoded constants to be iterated
-   balanced_generator: our balanced iterator
-   unrestricted_generator: the uniformly and independently sampling iterator
-   repeat: repeat generation of inside (values are sampled again)
-   zip: similar to Python `zip` function
-   cartesian_product: the cartesian product (two or more nested for-loops)

One can rewrite the previous code as follows:

```python
train_config = {
    "operation": "cartesian_product",
    "operands": [
        {
            "operation": "balanced_generator",
            "name": "key",
            "length": 16,
            "bunches": 1
        },
        {
            "operation": "balanced_generator",
            "name": "plaintext",
            "length": 16,
            "bunches": 1
        }
    ]
}
holdout_config = {
    "operation": "cartesian_product",
    "operands": [
        {
            "operation": "constants",
            "name": "key",
            "length": 1,
            "values": [
                # Single key with hardcoded values:
                [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ]
                # Alternatively we could pick a random key:
                #key = np.random.randint(low=0, high=256, size=16, dtype=np.int64)
            ]
        },
        {
            "operation": "balanced_generator",
            "name": "plaintext",
            "length": 16,
            "bunches": 100
        }
    ]
}

def capture_traces(config):
  """Single code for any split and iteration definition. Needs to be changed
  only when new masks are introduced.
  """
  for attack_points in build_attack_points_iterator(train_config):
    key = attack_points["key"]
    plaintext = attack_points["plaintext"]

    # ChipWhisperer takes bytearray as input.
    key = bytearray(key.astype(np.uint8))
    plaintext = bytearray(plaintext.astype(np.uint8))
    print(f"TODO capture with {key = } {plaintext = }")

# Capture train
capture_traces(train_config)

# Capture holdout (same code just the config is different)
capture_traces(holdout_config)
```
