#-*- coding: utf-8 -*-
import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

inputs = layers.Input(shape=x_train.shape[1:])

# block #1
net = layers.Conv2D(64, (3, 3), padding='same')(inputs)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2), strides=2, padding='same')(net)

# block #2
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2), strides=2, padding='same')(net)

# block #3
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.MaxPooling2D((2, 2), strides=2, padding='same')(net)

# block #4
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.MaxPooling2D((2, 2), strides=2, padding='same')(net)

# block #5
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)

net = layers.MaxPooling2D((2, 2), strides=2, padding='same')(net)

net = layers.GlobalAveragePooling2D()(net)

# softmax
net = layers.Dense(100, activation='softmax')(net)

model = keras.models.Model(inputs=inputs, outputs=net)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for i in range(5):
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('test loss:', loss, ', test accuracy:', accuracy)
