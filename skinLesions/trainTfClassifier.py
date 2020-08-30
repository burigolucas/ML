import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    # Data loaders for training, validation, and test sets
    PATH = 'data/'
    saved_model_path = 'model_transfer_MobileNetV2.h5'

    train_dir = os.path.join(PATH, 'train')
    valid_dir = os.path.join(PATH, 'valid')
    test_dir = os.path.join(PATH, 'test')

    BATCH_SIZE = 16
    IMG_SIZE = (160, 160)

    train_dataset = preprocessing.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    validation_dataset = preprocessing.image_dataset_from_directory(
        valid_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    test_dataset = preprocessing.image_dataset_from_directory(
        test_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    # image classes
    class_names = train_dataset.class_names

    # Using buffered prefetching

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    ])

    # Rescale pixel values using model-specific method
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet')

    # Freeze the convolutional base
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_names))

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    initial_epochs = 25

    checkpoint = ModelCheckpoint(saved_model_path, save_best_only=True)

    history = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
        callbacks=[checkpoint],
    )

    model.save(saved_model_path)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower left')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper left')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(saved_model_path+'.png')

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

if __name__ == '__main__':
    main()