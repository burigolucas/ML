import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
# EfficientNet
import efficientnet.tfkeras as efn
# import tensorflow.keras.applications.efficientnet as efn # requires tensorflow 2.3.0
# MobileNet
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import MobileNetV3Small
# from tensorflow.keras.applications import MobileNetV3Large

# from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE
MODELS = [
    efn.EfficientNetB0,
    efn.EfficientNetB1,
    efn.EfficientNetB2,
    efn.EfficientNetB3,
    efn.EfficientNetB4,
    efn.EfficientNetB5,
    efn.EfficientNetB6,
    MobileNet,
    MobileNetV2,
]

FINE_TUNING_LAYERS = {
    'efficientnet-b0': 'block7a',
    'efficientnet-b1': 'block7b',
    'efficientnet-b2': 'block7b',
    'efficientnet-b3': 'block7b',
    'efficientnet-b4': 'block7b',
    'efficientnet-b5': 'block7c',
    'efficientnet-b6': 'block7c',
    'efficientnetb0': 'block7a',
    'efficientnetb1': 'block7b',
    'efficientnetb2': 'block7b',
    'efficientnetb3': 'block7b',
    'efficientnetb4': 'block7b',
    'efficientnetb5': 'block7c',
    'efficientnetb6': 'block7c',
    'mobilenet_1.00_224': '_13',
    'mobilenetv2_1.00_224': 'block_16',
}


def _deserialize_example(example_proto, labeled=True):
    """
    Parse single example protocol buffer
    """
    if labeled:
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'patient_id': tf.io.FixedLenFeature([], tf.int64),
            'sex': tf.io.FixedLenFeature([], tf.int64),
            'age_approx': tf.io.FixedLenFeature([], tf.int64),
            'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
            'diagnosis': tf.io.FixedLenFeature([], tf.int64),
            'target': tf.io.FixedLenFeature([], tf.int64)
        }
    else:
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string)
        }

    return tf.io.parse_single_example(example_proto, feature_description)


def _select_features(parsed_example, config, augment, labeled=True):
    """
    Select features
    """

    img = _prepare_image(
        parsed_example['image'],
        config=config,
        augment=augment
    )
    if labeled:
        target = parsed_example['target']
        return img, target
    else:
        return img, 0


_augmentation_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(
        "horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(
        factor=(0., 0.25)),
    # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])


def _augment_image(img, config):
    """
    Augment images
    """

    # img = augmentations(img, training=True)
    # img = _transform_image(img,config)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_flip_up_down(img)
    img = tf.image.random_hue(img, 0.01)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.1)

    return img


def _prepare_image(img, config, augment):
    """
    Prepare images
    """

    dim = config['img_size']

    # Decode jpeg image
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    if augment:
        img = _augment_image(img, config)

    img = tf.reshape(img, [dim, dim, 3])

    if config['multiple_size']:
        dim1 = dim
        dim2 = int(dim/2)
        dim3 = int(dim/3)
        dim4 = int(dim/4)

        img2 = tf.image.resize(img, [dim2, dim2])
        img3 = tf.image.resize(img, [dim3, dim3])
        img4 = tf.image.resize(img, [dim4, dim4])
        img = (img, img2, img3, img4)

    return img


def read_dataset(files, config, augment=False, shuffle=False, labeled=True, return_image_names=False):
    """
    Read and deserialize the dataset from TFRecord files
    """
    image_names = []

    # retrieve raw dataset
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(1000)

    # parse raw dataset
    ds = ds.map(lambda x: _deserialize_example(
        x,
        labeled=labeled
    ), num_parallel_calls=AUTO)
    if return_image_names:
        image_names = ds.map(
            lambda parsed_example: parsed_example['image_name'])
    ds = ds.map(lambda x: _select_features(
        x,
        config=config,
        augment=augment,
        labeled=labeled
    ), num_parallel_calls=AUTO)

    ds = ds.batch(config['batch_size']*config['replicas'])
    # augment data using Keras Sequential model on batch
    if augment:
        if config['multiple_size']:
            ds = ds.map(lambda imgs, target: (
                (
                    _augmentation_model(imgs[0], training=True),
                    _augmentation_model(imgs[1], training=True),
                    _augmentation_model(imgs[2], training=True),
                    _augmentation_model(imgs[3], training=True)
                ), target), num_parallel_calls=AUTO)
        else:
            ds = ds.map(lambda img, target: (
                _augmentation_model(img, training=True), target
            ), num_parallel_calls=AUTO)

    ds = ds.prefetch(AUTO)

    return ds, image_names


def build_model(config):

    conv_base = MODELS[config['model_type']](
        input_shape=(None, None, 3),
        weights='imagenet',
        include_top=False)
    conv_base.trainable = False

    if config['multiple_size']:
        dim = config['img_size']

        dim1 = int(dim/1)
        dim2 = int(dim/2)
        dim3 = int(dim/3)
        dim4 = int(dim/4)

        inp1 = keras.layers.Input(shape=(dim1, dim1, 3))
        inp2 = keras.layers.Input(shape=(dim2, dim2, 3))
        inp3 = keras.layers.Input(shape=(dim3, dim3, 3))
        inp4 = keras.layers.Input(shape=(dim4, dim4, 3))

        inp = (inp1, inp2, inp3, inp4)

        x1 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inp1)
        x2 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inp2)
        x3 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inp3)
        x4 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inp4)

        x1 = conv_base(x1, training=False)
        x1 = keras.layers.GlobalAveragePooling2D()(x1)
        x2 = conv_base(x2, training=False)
        x2 = keras.layers.GlobalAveragePooling2D()(x2)
        x3 = conv_base(x3, training=False)
        x3 = keras.layers.GlobalAveragePooling2D()(x3)
        x4 = conv_base(x4, training=False)
        x4 = keras.layers.GlobalAveragePooling2D()(x4)

        x = keras.layers.concatenate([x1, x2, x3, x4])
        #x = tf.stack([x1,x2,x3,x4])
        #x = tf.reduce_sum(x,0)

    else:
        dim = config['img_size']
        inp = keras.layers.Input(shape=(dim, dim, 3))
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inp)

        x = conv_base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

    if config['dropout_rate'] > 0:
        x = keras.layers.Dropout(config['dropout_rate'])(x)
    output_bias = config['initial_bias']
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    out = keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=output_bias)(x)

    model = keras.Model(inputs=inp, outputs=out, name='CNN')
    model.summary()

    return conv_base, model


def compile_model(model, config):
    """
    Compile model for training
    """

    opt = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def learning_rate_callback(batch_size, replicas):
    """
    Train schedule for transfer learning by Chris Deotte
    """

    lr_start = 0.00005
    lr_max = 0.0000125 * replicas * batch_size
    lr_min = 0.00001
    lr_ramp_ep = 5
    lr_sus_ep = 5
    lr_decay = 0.8

    def lrfn(epoch):
        # warmup - linear increase
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        # plateau - max learning rate
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            # decay
            lr = (lr_max - lr_min) * lr_decay**(
                epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


def train(model, config, dataloader):
    """
    Train model
    """

    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=config['patience'],
        verbose=1,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        f"{config['model_label']}.h5",
        monitor='val_auc',
        mode='max',
        verbose=1,
        save_freq='epoch',
        save_best_only=True)

    # train
    history = model.fit(
        dataloader['train'],
        epochs=config['nb_epochs'],
        validation_data=dataloader['valid'],
        callbacks=[
            checkpoint,
            early_stopping,
            learning_rate_callback(
                batch_size=config['batch_size'],
                replicas=config['replicas'])
        ],
        class_weight=config['class_weight']
    )

    return history, model


def test(model, dataloader):
    """
    Test model
    """

    # Evaluate model on test dataset
    eval_metrics = model.evaluate(dataloader['test'], return_dict=True)

    return eval_metrics


def plot_history(history, config):
    """
    Plot training history to file
    """

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
    plt.ylim([0, 1.0])
    # plt.title('Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(train_prec, label='Training')
    plt.plot(val_prec, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('Precision')
    plt.ylim([0, 1.0])
    # plt.title('Training and Validation Precision')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 3)
    plt.plot(train_rec, label='Training')
    plt.plot(val_rec, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('Recall')
    plt.ylim([0, 1.0])
    # plt.title('Training and Validation Recall')

    plt.subplot(2, 2, 4)
    plt.plot(train_auc, label='Training')
    plt.plot(val_auc, label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('AUC')
    plt.ylim([0, 1.0])
    # plt.title('Training and Validation AUC')
    plt.xlabel('epoch')

    plt.savefig(f"{config['model_label']}.png")


def _get_transformation_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Returns 3x3 transformation matrix which transforms indicies
    """

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([
        c1, s1, zero, -s1, c1, zero, zero, zero, one
    ], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([
        one, s2, zero, zero, c2, zero, zero, zero, one
    ], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(tf.concat([
        one/height_zoom, zero, zero, zero, one/width_zoom, zero, zero, zero, one
    ], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([
        one, zero, height_shift, zero, one, width_shift, zero, zero, one
    ], axis=0), [3, 3])

    return keras.backend.dot(
        keras.backend.dot(rotation_matrix, shear_matrix),
        keras.backend.dot(zoom_matrix, shift_matrix))


def _transform_image(img, config):
    """
    Transform image

    input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    output - image randomly rotated, sheared, zoomed, and shifted
    """

    DIM = config['img_size']
    XDIM = DIM % 2  # fix for size 331

    rot = 15. * tf.random.uniform([1], maxval=90, dtype='float32')
    shr = 1. * tf.random.normal([1], dtype='float32')/10
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32')/20.
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32')/20.
    h_shift = 16. * tf.random.normal([1], dtype='float32')
    w_shift = 16. * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = _get_transformation_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = keras.backend.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = keras.backend.cast(idx2, dtype='int32')
    idx2 = keras.backend.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0, ], DIM//2-1+idx2[1, ]])
    img = tf.gather_nd(img, tf.transpose(idx3))

    return tf.reshape(img, [DIM, DIM, 3])


def get_strategy(device):
    """
    Return device-specific strategy
    """

    if device == "TPU":
        print("[INFO] Connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('[INFO] Running on TPU ', tpu.master())
        except ValueError:
            print("[INFO] Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("[INFO] Initializing TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("[INFO] TPU initialized")
            except:
                print("[INFO] Failed to initialize TPU")
        else:
            device = "GPU"

    if device != "TPU":
        print("[INFO] Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()

    if device == "GPU":
        print("[INFO] Num GPUs Available: ", len(
            tf.config.experimental.list_physical_devices('GPU')))

    return strategy


def enable_fine_tunning(conv_base, config):
    """
    Turn on fine tuning in CNN base
    """

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if FINE_TUNING_LAYERS[conv_base.name] in layer.name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
            print(f"[INFO] Setting {layer.name} trainable")
        else:
            layer.trainable = False


def create_submission_file(image_names, preds, outputfile='submission.csv'):

    image_names = np.array([img_name.numpy().decode("utf-8")
                            for img_name in iter(image_names)])

    preds = np.array(preds).mean(axis=0)

    submission = pd.DataFrame(dict(image_name=image_names, target=preds))
    submission = submission.sort_values('image_name')
    submission.to_csv(outputfile, index=False)


def main():

    tic = time.perf_counter()

    DEVICE = 'GPU'
    SEED = 42
    IMG_SIZE = 128
    MULTIPLE_IMG_SIZE = False
    BATCH_SIZE = 32
    FOLDS = 3
    AUGMENT = False
    FINE_TUNING = False

    strategy = get_strategy(DEVICE)
    REPLICAS = 1
    REPLICAS = strategy.num_replicas_in_sync
    print(f'[INFO] REPLICAS: {REPLICAS}')

    # oof_pred = []; oof_tar = []; oof_val = []; oof_names = []; oof_folds = []
    # preds = np.zeros((count_data_items(files_test),1))

    # Model type
    # 0-6: EfficientNet B0-B6
    # 7-8: MobileNet V1, V2
    MODEL_TYPE = 7

    PATH = f"./data/ISIC2020-{IMG_SIZE}x{IMG_SIZE}"
    # GCS_PATH1 = KaggleDatasets().get_gcs_path(f'melanoma-{IMG_SIZE}x{IMG_SIZE}')
    # GCS_PATH2 = KaggleDatasets().get_gcs_path(f'isic2019-{IMG_SIZE}x{IMG_SIZE}')

    skf = KFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    )

    oof_val = []
    settings = []
    preds = []

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):

        print(f'[INFO] Initializing fold {fold}')

        config = {
            'img_size': IMG_SIZE,
            'multiple_size': MULTIPLE_IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'replicas': REPLICAS,
            'model_label': f'model_{MODEL_TYPE}_{IMG_SIZE}_CV{FOLDS}_fold{fold}',
            'model_type': MODEL_TYPE,
            'nb_epochs': 15,
            'patience': 5,
            'initial_bias': None,
            'class_weight': None,
            'augment': AUGMENT,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob(
            [PATH + '/train%.2i*.tfrec' % x for x in idxT])
        # if INC2019:
        #     files_train += tf.io.gfile.glob([GCS_PATH2 + '/train%.2i*.tfrec'%x for x in idxT*2+1])
        #     print('#### Using 2019 external data')
        np.random.shuffle(files_train)

        files_valid = tf.io.gfile.glob(
            [PATH + '/train%.2i*.tfrec' % x for x in idxV])

        files_test = np.sort(
            np.array(tf.io.gfile.glob(PATH + '/test*.tfrec'))).tolist()

        # files_train, files_valid = train_test_split(
        #     files_train, test_size = 0.2, random_state = SEED)

        # print(f"Files train: {len(files_train)}")
        # print(f"Files valid: {len(files_valid)}")
        # print(f"Files test: {len(files_test)}")

        ds_train, _ = read_dataset(
            files=files_train,
            config=config,
            augment=config['augment'],
            shuffle=True
        )
        ds_valid, _ = read_dataset(
            files=files_valid,
            config=config,
            augment=False
        )
        ds_test, image_names = read_dataset(
            files=files_test,
            config=config,
            augment=False,
            labeled=False,
            return_image_names=True
        )

        dataloader = {
            'train': ds_train,
            'valid': ds_valid,
            'test': ds_test
        }

        # Handling imbalance data
        # check: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        pos = 0
        tot = 0
        for _, labels in ds_train:
            tot += labels.shape[0]
            pos += labels.numpy().sum()
        neg = tot - pos
        print(f'[INFO] Class imbalance: {pos/tot:f}')
        config['initial_bias'] = np.log([pos/neg]).tolist()

        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg)*(tot)/2.0
        weight_for_1 = (1 / pos)*(tot)/2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}
        config['class_weight'] = class_weight

        print('[INFO] Weight for class 0: {:.2f}'.format(weight_for_0))
        print('[INFO] Weight for class 1: {:.2f}'.format(weight_for_1))

        # def benchmark(dataset, num_epochs=12):
        #     start_time = time.perf_counter()
        #     for epoch_num in range(num_epochs):
        #         for sample in dataset:
        #             # Performing a training step
        #             time.sleep(0.01)
        #     tf.print("Execution time:", time.perf_counter() - start_time)

        # benchmark(ds_train)

        # for batch in ds_train.take(1):
        #     print(batch['sex'][0].numpy())

        #     img = batch['image'][0]
        #     print(img.shape)
        #     plt.imshow(img)
        #     plt.show()

        # clear session when building models in a loop
        tf.keras.backend.clear_session()

        with strategy.scope():
            conv_base, model = build_model(config=config)
            model = compile_model(model, config)

            history, model = train(
                model=model,
                dataloader=dataloader,
                config=config)
            history_fit = history.history
            history_fit['lr'] = [float(lr) for lr in history_fit['lr']]

            if FINE_TUNING:
                config['learning_rate'] = config['learning_rate']/10
                config['nb_epochs'] = 5
                config['model_label'] = f"{config['model_label']}_fineTunned"
                enable_fine_tunning(conv_base, config)
                model = compile_model(model, config)
                history_fine, model = train(
                    model=model,
                    dataloader=dataloader,
                    config=config)
                history_fine = history_fine.history
                history_fine['lr'] = [float(lr) for lr in history_fine['lr']]

            print('Making predictions...')
            model.load_weights(f"{config['model_label']}.h5")
            predictions = model.predict(dataloader['test'])
            preds.append(predictions.squeeze())

        # save settings to json file
        fold_settings = {
            'config': config,
            'history': history_fit,
            'history_fine': history_fine if FINE_TUNING else None,
            'files': {
                'train': files_train,
                'valid': files_valid,
                # 'test': files_test
            }
        }
        settings.append(fold_settings)
        json.dump(settings, open(
            f"model_{MODEL_TYPE}_{IMG_SIZE}_CV{FOLDS}_results.json", 'w'))

        # Report out of fold validation results
        if FINE_TUNING:
            oof_val.append(
                np.max(history_fit['val_auc'] + history_fine['val_auc']))
        else:
            oof_val.append(np.max(history_fit['val_auc']))
        print(f'[INFO] Fold {fold} - OOF AUC = {oof_val[-1]:.3f}')

    print(f"[INFO] IMG_SIZE {IMG_SIZE} - MULTIPLE_IMG_SIZE {MULTIPLE_IMG_SIZE:d} - AUGMENT {AUGMENT:d} - FINE_TUNING {FINE_TUNING:d} - LR {config['learning_rate']}")
    print(
        f'[INFO] MODEL {conv_base.name} with image Size {IMG_SIZE} - Mean OOF AUC: {np.mean(oof_val):.4f}')
    ix_max = np.argmax(oof_val)
    print(f"[INFO] Max OOF AUC {oof_val[ix_max]:.4f} for fold {ix_max}")
    json.dump(
        {'settings': settings, 'oof_val': oof_val},
        open(f"model_{MODEL_TYPE}_{IMG_SIZE}_CV{FOLDS}_results.json", 'w')
    )

    toc = time.perf_counter()
    print(f"[INFO] TIME: {toc - tic:0.4f} seconds")
    create_submission_file(image_names, preds)


if __name__ == '__main__':
    main()
