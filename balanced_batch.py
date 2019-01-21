import random
import numpy as np

def balanced_batch(x, y, batch_size, categorical_binary=False):
    if x.shape[0] != y.shape[0]:
        raise ValueError('Args `x` and `y` should have the same length.')
    if len(y.shape) != 2:
        raise ValueError(
            'Arg `y` should have a shape of (num_samples, num_classes). ' +
            'Use keras.utils.to_categorical to convert a class vector ' +
            'to a binary class matrix.'
        )
    num_samples = y.shape[0]
    num_classes = y.shape[1]
    binary = num_classes == 2 and not categorical_binary
    batch_x_shape = (batch_size, *x.shape[1:])
    batch_y_shape = (batch_size, ) if binary else (batch_size, num_classes)
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
            if binary:
                batch_y[i] = random_class
            else:
                batch_y[i][random_class] = 1
        yield (batch_x, batch_y)
