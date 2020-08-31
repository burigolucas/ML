import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(model_label,nb_classes):

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    ])

    if model_label.lower() == 'resnet50':
        IMG_SIZE = (224, 224)
        IMG_SHAPE = IMG_SIZE + (3,)

        # Rescale pixel values using model-specific method
        preprocess_input = tf.keras.applications.resnet50.preprocess_input

        # Create the base model from the pre-trained model ResNet50
        base_model = tf.keras.applications.ResNet50(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )

    else:
        IMG_SIZE = (160, 160)
        IMG_SHAPE = IMG_SIZE + (3,)

        # Rescale pixel values using model-specific method
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )

    # Freeze the convolutional base
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(nb_classes)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model    

def main():
    # Define settings
    BATCH_SIZE = 16
    nb_epochs = 25
    learning_rate = 0.001
    data_dir = 'data/'
    model_label = 'ResNet50'

    saved_model_path = f'model_transfer_{model_label}.h5'

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    if model_label.lower() == 'resnet50':
        IMG_SIZE = (224, 224)
    else:
        IMG_SIZE = (160, 160)

    # Data loaders for training, validation, and test sets
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

    model = build_model(model_label=model_label,nb_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(saved_model_path, save_best_only=True)

    history = model.fit(
        train_dataset,
        epochs=nb_epochs,
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
    plt.savefig(f'fig_history_{model_label}.png')

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

if __name__ == '__main__':
    main()