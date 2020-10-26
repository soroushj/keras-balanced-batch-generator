# Balanced Batch Generator

An over-sampling algorithm to address the issue of class imbalance.

Generates *balanced batches*, i.e., batches in which the number of samples from
each class is on average the same. Generated batches are also shuffled.

Can be easily used with Keras models'
[`fit`](https://keras.io/api/models/model_training_apis/#fit-method).

Currently, only NumPy arrays for single-input, single-output models are supported.

## API

```python
balanced_batch_generator(x, y, batch_size,
                         categorical=True,
                         seed=None)
```

- **`x`** *(numpy.ndarray)* Input data. Must have the same length as `y`.
- **`y`** *(numpy.ndarray)* Target data. Must be a binary class matrix (i.e.,
  shape `(num_samples, num_classes)`). You can use `keras.utils.to_categorical`
  to convert a class vector to a binary class matrix.
- **`batch_size`** *(int)* Batch size.
- **`categorical`** *(bool, optional)* If true, generates binary class matrices
  (i.e., shape `(num_samples, num_classes)`) for batch targets.
  Otherwise, generates class vectors (i.e., shape `(num_samples, )`).
  Defaults to `True`.
- **`seed`** *(int, optional)* Random seed (see the [docs](https://docs.python.org/3/library/random.html#random.seed)).
- Returns a generator yielding batches as tuples `(x, y)` that can
  be directly used with Keras.

## Basic Usage

```python
from keras.models import Sequential
from balanced_batch_generator import balanced_batch_generator

x = ...
y = ...
batch_size = ...
steps_per_epoch = ...
model = Sequential(...)

batch_generator = balanced_batch_generator(x, y, batch_size)
model.fit(batch_generator, steps_per_epoch=steps_per_epoch)
```

## Example: Multiclass Classification

```python
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from balanced_batch_generator import balanced_batch_generator

num_samples = 100
num_classes = 3
input_shape = (2, )
batch_size = 16

x = np.random.rand(num_samples, *input_shape)
y = np.random.randint(low=0, high=num_classes, size=num_samples)
y = to_categorical(y)

batch_generator = balanced_batch_generator(x, y, batch_size)

model = Sequential()
model.add(Dense(32, input_shape=input_shape, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(batch_generator, steps_per_epoch=10, epochs=5)
```

## Example: Binary Classification

```python
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from balanced_batch_generator import balanced_batch_generator

num_samples = 100
num_classes = 2
input_shape = (2, )
batch_size = 16

x = np.random.rand(num_samples, *input_shape)
y = np.random.randint(low=0, high=num_classes, size=num_samples)
y = to_categorical(y)

batch_generator = balanced_batch_generator(x, y, batch_size, categorical=False)

model = Sequential()
model.add(Dense(32, input_shape=input_shape, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(batch_generator, steps_per_epoch=10, epochs=5)
```
