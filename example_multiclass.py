import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras_balanced_batch_generator import make_generator

def example_multiclass():
    num_samples = 100
    num_classes = 3
    input_shape = (2, )
    batch_size = 16

    x = np.random.rand(num_samples, *input_shape)
    y = np.random.randint(low=0, high=num_classes, size=num_samples)
    y = to_categorical(y)

    generator = make_generator(x, y, batch_size)

    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(generator, steps_per_epoch=10, epochs=5)

if __name__ == '__main__':
    example_multiclass()
