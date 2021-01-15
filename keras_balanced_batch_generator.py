import random
import numpy as np

def make_generator(x, y, batch_size,
                   categorical=True,
                   seed=None):
    """A Keras-compatible generator for creating balanced batches.

    This generator loops over its data indefinitely and yields balanced,
    shuffled batches.

    Args:
    x (numpy.ndarray): Input data. Must have the same length as `y`.
    y (numpy.ndarray): Target data. Must be a binary class matrix (i.e.,
        shape `(num_samples, num_classes)`). You can use
        `keras.utils.to_categorical` to convert a class vector to a binary
        class matrix.
    batch_size (int): Batch size.
    categorical (bool): If true, generates binary class matrices
        (i.e., shape `(num_samples, num_classes)`) for batch targets.
        Otherwise, generates class vectors (i.e., shape `(num_samples,)`).
    seed: Random seed.
    Returns a Keras-compatible generator yielding batches as `(x, y)` tuples.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Args `x` and `y` must have the same length.')
    if x.shape[0] < 1:
        raise ValueError('Args `x` and `y` must not be empty.')
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
    batch_y_shape = (batch_size, num_classes) if categorical else (batch_size,)
    indexes = [0 for _ in range(num_classes)]
    samples = [[] for _ in range(num_classes)]
    for i in range(num_samples):
        samples[np.argmax(y[i])].append(x[i])
    for c, s in enumerate(samples):
        if len(s) < 1:
            raise ValueError('Class {} has no samples.'.format(c))
    rand = random.Random(seed)
    while True:
        batch_x = np.ndarray(shape=batch_x_shape, dtype=x.dtype)
        batch_y = np.zeros(shape=batch_y_shape, dtype=y.dtype)
        for i in range(batch_size):
            random_class = rand.randrange(num_classes)
            current_index = indexes[random_class]
            indexes[random_class] = (current_index + 1) % len(samples[random_class])
            if current_index == 0:
                rand.shuffle(samples[random_class])
            batch_x[i] = samples[random_class][current_index]
            if categorical:
                batch_y[i][random_class] = 1
            else:
                batch_y[i] = random_class
        yield (batch_x, batch_y)
