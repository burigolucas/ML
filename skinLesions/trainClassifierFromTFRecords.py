import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn
AUTO     = tf.data.experimental.AUTOTUNE
MODELS = [
    efn.EfficientNetB0,
    efn.EfficientNetB1,
    efn.EfficientNetB2,
    efn.EfficientNetB3,
    efn.EfficientNetB4,
    efn.EfficientNetB5,
    efn.EfficientNetB6
]

def _deserialize_example(example_proto):
    """
    Parse single example protocol buffer
    """

    feature_description = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }
 
    return tf.io.parse_single_example(example_proto, feature_description)

def _select_features(parsed_example,config,augment):
    """
    Select features
    """
    img = _prepare_image(parsed_example['image'],config=config,augment=augment)
    target = parsed_example['target']
                         
    return img, target

def _augment_image(img, config):

    return img

def _prepare_image(img, config, augment):

    dim = config['img_size']

    # Decode jpeg image
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = _augment_image(img, config)
                      
    img = tf.reshape(img, [dim, dim, 3])

    if config['multiple_size']:
        dim1 = dim
        dim2 = int(dim/2)
        dim3 = int(dim/3)
        dim4 = int(dim/4)
        
        img2 = tf.image.resize(img,[dim2,dim2])
        img3 = tf.image.resize(img,[dim3,dim3])
        img4 = tf.image.resize(img,[dim4,dim4])
        img = (img,img2,img3,img4)

    return img

def read_dataset(files,config,augment=False):
    """
    Read and deserialize the dataset from TFRecord files
    """
    # retrieve raw dataset
    ds  = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    # parse raw dataset
    ds = ds.map(_deserialize_example, num_parallel_calls=AUTO)
    ds = ds.map(lambda x: _select_features(x, config=config, augment=augment), num_parallel_calls=AUTO)

    ds = ds.batch(config['batch_size']*config['replicas'])

    ds = ds.prefetch(AUTO)

    return ds
    
def build_model(config):

    conv_base = MODELS[config['model_type']](
        input_shape=(None,None,3),
        weights='imagenet',
        include_top=False)
    conv_base.trainable = False

    if config['multiple_size']:
        dim1 = config['img_size']
        dim2 = int(dim1/2)
        dim3 = int(dim1/3)
        dim4 = int(dim1/4)

        inp1 = keras.layers.Input(shape=(dim1,dim1,3))
        inp2 = keras.layers.Input(shape=(dim2,dim2,3))
        inp3 = keras.layers.Input(shape=(dim3,dim3,3))
        inp4 = keras.layers.Input(shape=(dim4,dim4,3))
        inp = (inp1,inp2,inp3,inp4)

        x1 = conv_base(inp1, training=False)
        x1 = keras.layers.GlobalAveragePooling2D()(x1)
        x2 = conv_base(inp2, training=False)
        x2 = keras.layers.GlobalAveragePooling2D()(x2)
        x3 = conv_base(inp3, training=False)
        x3 = keras.layers.GlobalAveragePooling2D()(x3)
        x4 = conv_base(inp4, training=False)
        x4 = keras.layers.GlobalAveragePooling2D()(x4)
        x = keras.layers.concatenate([x1,x2,x3,x4])
    else:
        dim = config['img_size']
        inp = keras.layers.Input(shape=(dim,dim,3))
        x = conv_base(inp, training=False)

    if config['dropout_rate'] > 0:
        x = keras.layers.Dropout(config['dropout_rate'])(x)
    output_bias = config['initial_bias']
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    x = keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer = output_bias)(x)

    model = keras.Model(inputs=inp,outputs=x)

    return conv_base, model

def train(model,config,dataloader):

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

    model.compile(optimizer=opt,loss=loss,metrics=metrics)

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
        callbacks=[checkpoint,early_stopping],
        class_weight=config['class_weight']
    )

    return history, model

def test(model,dataloader):
    return True

def main():

    SEED = 42
    IMG_SIZE = 192
    BATCH_SIZE = 64
    REPLICAS = 1
    # EfficientNet type
    MODEL_TYPE = 0
    config = {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'replicas': REPLICAS,
        'model_label': f'model_B{MODEL_TYPE}_{IMG_SIZE}_CV{FOLDS}_fold{fold}',
        'model_type': MODEL_TYPE,
        'nb_epochs': 12,
        'patience': 5,
        'initial_bias': None,
        'class_weight': None,
        'augment': False,
        'multiple_size': True,
        'dropout_rate': 0.2,
    }

    PATH = f"/data/melanoma/{IMG_SIZE}x{IMG_SIZE}"

    files = np.sort(np.array(tf.io.gfile.glob(PATH + '/train*.tfrec')))

    files_train, files_test = train_test_split(
        files, test_size = 0.2, random_state = SEED)

    files_train, files_valid = train_test_split(
        files_train, test_size = 0.2, random_state = SEED)

    print(f"Files train: {len(files_train)}")
    print(f"Files valid: {len(files_valid)}")
    print(f"Files test: {len(files_test)}")

    ds_train = read_dataset(
        files=files_train,
        config=config)
    ds_valid = read_dataset(
        files=files_valid,
        config=config)
    ds_test  = read_dataset(
        files=files_test,
        config=config)

    dataloader = {
        'train': ds_train,
        'valid': ds_valid,
        'test': ds_test
    }

    conv_base, model = build_model(config = config)
    model = train(
        model = model,
        dataloader = dataloader,
        config = config)

    test(model,dataloader)

if __name__ == '__main__':
    main()