---
title: Capture Resume
description: Resuming a Crashed Capture
---

A capture campaign can easily take anywhere from a couple of minutes to several months.
A crash may occur for many reasons such as a power outage.
When a crash happens the generated inputs by attack point iterators would not be completely balanced.
We build our tooling to be resistant to crashes and thus resumable.
We first save all inputs of the algorithm of interest and then iterate those while automatically saving progress.

# Resume Capture Overview

First we save all we create all input values.
Save those.
Then we load and iterate and in case of an error or a crash we just reload where we stopped.
Let us go through the code with the concrete case of AES128.

### Save Inputs

In the last guide we created iterators over inputs.
We can accumulate all values in arrays `keys` and `plaintexts` where `keys.shape = (65_536, 16)` and `plaintexts.shape = (65_536, 16`.

```python
from scaaml.io.resume_kti import create_resume_kti

# Save all encryption parameters ahead before the capture campaign starts.
# If `kt_filename` exists this does nothing.
create_resume_kti(
  parameters={"keys": keys, "texts": texts},
  shard_length=32,
  kt_filename="parameters_tuples.txt",
  progress_filename="progress_tuples.txt",
)
```

- We save measurements in files called shards.
  Each of which contains `shard_length` examples.
  An error could happen while saving the file.
  We thus restart at the lowest multiple of `shard_length` to ensure that data have been saved.
  One can set `shard_length=1` so that we do not repeat successful captures.
- Note that it is not necessary for the `keys` and `plaintexts` to have the same length.
  The only condition we require is that there is the same number of those.
  That is we need to `assert keys.shape[0] == plaintexts.shape[0]`.
- If we wanted to save more values in `parameters` we could.
  The only condition is that all of them have the same `len`.

### Load the Saved Inputs

```python
from scaaml.io.resume_kti import ResumeKTI

# Load the save point (all the tuples + the index where to continue) at the
# start or resume of the experiment.
resume_kti = ResumeKTI(
  kt_filename="parameters_tuples.txt",
  progress_filename="progress_tuples.txt",
)
```

If necessary one can manually change the value in `progress_tuples` before loading (see section File Format).

### Iterate

Now we can iterate `namedtuple`s we have saved previously.

```python
from tqdm import tqdm

# Loop with a progress-bar (progress bar has always the same length, but when
# restarted the progress is not lost). One could avoid having the progress-bar
# by `for current_parameters in resume_kti:`
for key, text in tqdm(resume_kti, initial=resume_kti.initial_index):
  print(f"TODO capture with {key = } {plaintext = }")
  # An error can happen when capturing or saving
```

Note that the names in the namedtuple are the same as in the parameters.
Thus if you use "keys" for an array of keys then you get plural even for
the single iterated key.
`current_parameters = { "keys": keys[i], "texts": texts[i] }`

## File Format

Parameter tuples file is a binary file with numpy scalar and arrays.  The
contents are shard length, array of strings with parameter names and then
arrays of parameters in the same order.  The array of parameter values are
interpreted in the way that (`parameter1[i], parameter2[i], ...`) is a single
key-text tuple for any `i in range(len(parameter1))`, thus all have the same
length.  This is done in order to allow different data type of key and text
(for instance AES256 where key is 256 bits) and different number of saved
parameters (saving masks...).

Progress file is a text file with a single integer -- the number of traces
captured so far. The number in it is a multiple of the shard length. This file
is automatically updated after shard length iterations over ResumeKTI.
