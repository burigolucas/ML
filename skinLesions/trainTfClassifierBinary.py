import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(model_label):

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
    prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')

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
    learning_rate = 0.0001
    data_dir = 'dataset/'
    model_label = 'ResNet50'

    saved_model_path = f'modelBinary_transfer_{model_label}.h5'

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
        follow_links=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    validation_dataset = preprocessing.image_dataset_from_directory(
        valid_dir,
        shuffle=True,
        follow_links=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    test_dataset = preprocessing.image_dataset_from_directory(
        test_dir,
        shuffle=True,
        follow_links=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE)

    # image classes
    class_names = train_dataset.class_names
    nbClasses = len(class_names)
    assert nbClasses == 2
    print(f"Classes: {class_names}")

    # Using buffered prefetching
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    model = build_model(model_label=model_label)

    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(saved_model_path, save_best_only=True)

    history = model.fit(
        train_dataset,
        epochs=nb_epochs,
        validation_data=validation_dataset,
        callbacks=[checkpoint,early_stopping],
    )

    model.save(saved_model_path)

    # Plot training and validation history
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_prec = history.history['precision']
    val_prec = history.history['val_precision']
    train_rec = history.history['recall']
    val_rec = history.history['val_recall']
    train_auc = history.history['auc']
    val_auc = history.history['val_auc']
      
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_acc, label='Training')
    plt.plot(val_acc, label='Validation')
    plt.legend(loc='lower left')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    #plt.title('Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(train_prec, label='Training')
    plt.plot(val_prec, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('Precision')
    plt.ylim([0,1.0])
    #plt.title('Training and Validation Precision')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 3)
    plt.plot(train_rec, label='Training')
    plt.plot(val_rec, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('Recall')
    plt.ylim([0,1.0])
    #plt.title('Training and Validation Recall')

    plt.subplot(2, 2, 4)
    plt.plot(train_auc, label='Training')
    plt.plot(val_auc, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('AUC')
    plt.ylim([0,1.0])
    #plt.title('Training and Validation AUC')
    plt.xlabel('epoch')

    plt.savefig(f'fig_history_{model_label}.png')

    # Evaluate model on test dataset
    eval_metrics = model.evaluate(test_dataset,return_dict=True)
    print(f'Evaluation metrics : {eval_metrics}')

if __name__ == '__main__':
    main()