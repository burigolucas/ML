import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn
AUTO     = tf.data.experimental.AUTOTUNE

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

def _select_features(parsed_example,config):
    """
    Select features
    """
    img = _prepare_image(parsed_example['image'],config=config)
    target = parsed_example['target']
                         
    return img, target

def _prepare_image(img, config):

    # Decode jpeg image
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    img = tf.reshape(img, [config['img_size'],config['img_size'], 3])
    return img

def read_dataset(files,config):
    """
    Read and deserialize the dataset from TFRecord files
    """
    # retrieve raw dataset
    ds  = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    # parse raw dataset
    ds = ds.map(_deserialize_example, num_parallel_calls=AUTO)
    ds = ds.map(lambda x: _select_features(x, config=config), num_parallel_calls=AUTO)

    ds = ds.batch(config['batch_size']*config['replicas'])

    ds = ds.prefetch(AUTO)

    return ds
    
def build_model(config):

    dim = config['img_size']

    output_bias = config['initial_bias']
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inp = keras.layers.Input(shape=(dim,dim,3))
    conv_base = efn.EfficientNetB4(input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    conv_base.trainable = False
    x = conv_base(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer = output_bias)(x)
    model = keras.Model(inputs=inp,outputs=x)

    return model

def train(model,config,dataloader):
    return model

def test(model,dataloader):
    return True

def main():

    SEED = 42
    IMG_SIZE = 192
    BATCH_SIZE = 64
    REPLICAS = 1
    config = {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'replicas': REPLICAS,
        'saved_model_path': 'model_EfficientNetB4fromTFRecords.h5',
        'nb_epochs': 12,
        'patience': 5,
        'initial_bias': None,
        'class_weight': None
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

    model = build_model(config = config)
    model = train(
        model = model,
        dataloader = dataloader,
        config = config)

    test(model,dataloader)

if __name__ == '__main__':
    main()