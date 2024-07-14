import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
class Baseline(tf.keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.f2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

# show
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()