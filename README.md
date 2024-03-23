# keras-balanced-batch-generator: A Keras-compatible generator for creating balanced batches

[![PyPI](https://img.shields.io/pypi/v/keras-balanced-batch-generator.svg)](https://pypi.org/project/keras-balanced-batch-generator/)
[![MIT license](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install keras-balanced-batch-generator
```

## Overview

This module implements an over-sampling algorithm to address the issue of class imbalance.
It generates *balanced batches*, i.e., batches in which the number of samples from each class is on average the same.
Generated batches are also shuffled.

The generator can be easily used with Keras models'
[`fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) method.

Currently, only [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) for single-input, single-output models are supported.

## API

```python
make_generator(x, y, batch_size,
               categorical=True,
               seed=None)
```

- **`x`** *(numpy.ndarray)* Input data. Must have the same length as `y`.
- **`y`** *(numpy.ndarray)* Target data. Must be a binary class matrix (i.e., shape `(num_samples, num_classes)`).
  You can use [`keras.utils.to_categorical`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical) to convert a class vector to a binary class matrix.
- **`batch_size`** *(int)* Batch size.
- **`categorical`** *(bool)* If true, generates binary class matrices (i.e., shape `(num_samples, num_classes)`) for batch targets.
  Otherwise, generates class vectors (i.e., shape `(num_samples,)`).
- **`seed`** Random seed (see the [docs](https://docs.python.org/3/library/random.html#random.seed)).
- Returns a Keras-compatible generator yielding batches as `(x, y)` tuples.

## Usage

```python
import keras
from keras_balanced_batch_generator import make_generator

x = ...
y = ...
batch_size = ...
steps_per_epoch = ...
model = keras.models.Sequential(...)

generator = make_generator(x, y, batch_size)
model.fit(generator, steps_per_epoch=steps_per_epoch)
```

## Example: Multiclass Classification

```python
import numpy as np
import keras
from keras_balanced_batch_generator import make_generator

num_samples = 100
num_classes = 3
input_shape = (2,)
batch_size = 16

x = np.random.rand(num_samples, *input_shape)
y = np.random.randint(low=0, high=num_classes, size=num_samples)
y = keras.utils.to_categorical(y)

generator = make_generator(x, y, batch_size)

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_shape=input_shape, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(generator, steps_per_epoch=10, epochs=5)
```

## Example: Binary Classification

```python
import numpy as np
import keras
from keras_balanced_batch_generator import make_generator

num_samples = 100
num_classes = 2
input_shape = (2,)
batch_size = 16

x = np.random.rand(num_samples, *input_shape)
y = np.random.randint(low=0, high=num_classes, size=num_samples)
y = keras.utils.to_categorical(y)

generator = make_generator(x, y, batch_size, categorical=False)

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_shape=input_shape, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(generator, steps_per_epoch=10, epochs=5)
```
