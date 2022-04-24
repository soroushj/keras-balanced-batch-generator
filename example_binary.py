import numpy as np
import tensorflow as tf
from keras_balanced_batch_generator import make_generator

def example_binary():
    num_samples = 100
    num_classes = 2
    input_shape = (2,)
    batch_size = 16

    x = np.random.rand(num_samples, *input_shape)
    y = np.random.randint(low=0, high=num_classes, size=num_samples)
    y = tf.keras.utils.to_categorical(y)

    generator = make_generator(x, y, batch_size, categorical=False)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(generator, steps_per_epoch=10, epochs=5)

if __name__ == '__main__':
    example_binary()
