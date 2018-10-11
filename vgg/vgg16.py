#-*- coding: utf-8 -*-
import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

inputs = layers.Input(shape=x_train.shape[1:])

blocks = [[64] * 2, [128] * 2, [256] * 3, [512] * 3, [512] * 3]

net = inputs
for block in blocks:
    for filters in block:
        net = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(net)
    net = layers.MaxPooling2D((2, 2), strides=2)(net)

net = layers.Flatten()(net)
net = layers.Dense(4096, activation='relu')(net)
net = layers.Dense(4096, activation='relu')(net)
net = layers.Dense(100, activation='softmax')(net)


model = keras.models.Model(inputs=inputs, outputs=net)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for i in range(5):
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('test loss:', loss, ', test accuracy:', accuracy)
