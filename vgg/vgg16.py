#-*- coding: utf-8 -*-
import keras
from keras import layers


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# x_train /= 255.
# x_test /= 255.

inputs = layers.Input(shape=x_train.shape[1:])
# inputs = layers.Input(shape=(224, 224, 3))

blocks = [[64] * 2, [128] * 2, [256] * 3, [512] * 3, [512] * 3]

net = inputs
for block in blocks:
    for filters in block:
        net = layers.Conv2D(filters, (3, 3), padding='same')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dropout(0.4)(net)
    net = layers.MaxPooling2D((2, 2), strides=2)(net)

net = layers.Flatten()(net)
net = layers.Dense(1024)(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)

net = layers.Dense(100, activation='softmax')(net)


model = keras.models.Model(inputs=inputs, outputs=net)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

for i in range(5):
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('test loss:', loss, ', test accuracy:', accuracy)
