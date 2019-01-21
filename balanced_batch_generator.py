import random
import numpy as np

def balanced_batch_generator(x, y, batch_size, categorical=True):
    """A generator for creating balanced batched.

    This generator loops over its data indefinitely and yields balanced,
    shuffled batches.

    Args:
    x (numpy.ndarray): Samples (inputs). Must have the same length as `y`.
    y (numpy.ndarray): Labels (targets). Must be a binary class matrix (i.e.,
        shape `(num_samples, num_classes)`). You can use `keras.utils.to_categorical`
        to convert a class vector to a binary class matrix.
    batch_size (int): Batch size.
    categorical (bool, optional): If true, generates binary class matrices
        (i.e., shape `(num_samples, num_classes)`) for batch labels (targets).
        Otherwise, generates class vectors (i.e., shape `(num_samples, )`).
        Defaults to `True`.
    Returns a generator yielding batches as tuples `(inputs, targets)` that can
        be directly used with Keras.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Args `x` and `y` must have the same length.')
    if len(y.shape) != 2:
        raise ValueError(
            'Arg `y` must have a shape of (num_samples, num_classes). ' +
            'You can use `keras.utils.to_categorical` to convert a class vector ' +
            'to a binary class matrix.'
        )
    if batch_size < 1:
        raise ValueError('Arg `batch_size` must be a positive integer.')
    num_samples = y.shape[0]
    num_classes = y.shape[1]
    batch_x_shape = (batch_size, *x.shape[1:])
    batch_y_shape = (batch_size, num_classes) if categorical else (batch_size, )
    indexes = [0 for _ in range(num_classes)]
    samples = [[] for _ in range(num_classes)]
    for i in range(num_samples):
        samples[np.argmax(y[i])].append(x[i])
    while True:
        batch_x = np.ndarray(shape=batch_x_shape, dtype=x.dtype)
        batch_y = np.zeros(shape=batch_y_shape, dtype=y.dtype)
        for i in range(batch_size):
            random_class = random.randrange(num_classes)
            current_index = indexes[random_class]
            indexes[random_class] = (current_index + 1) % len(samples[random_class])
            if current_index == 0:
                random.shuffle(samples[random_class])
            batch_x[i] = samples[random_class][current_index]
            if categorical:
                batch_y[i][random_class] = 1
            else:
                batch_y[i] = random_class
        yield (batch_x, batch_y)
