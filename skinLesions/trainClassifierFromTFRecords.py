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

def _select_features(parsed_example):
    """
    Select features
    """
    # Decode jpeg image
    parsed_example['image'] = tf.image.decode_jpeg(parsed_example['image'], channels=3)

    return parsed_example['image'],parsed_example['target']

def read_dataset(files,batch_size):
    """
    Read and deserialize the dataset from TFRecord files
    """
    # retrieve raw dataset
    ds  = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    # parse raw dataset
    ds = ds.map(_deserialize_example, num_parallel_calls=AUTO)
    ds = ds.map(_select_features, num_parallel_calls=AUTO)

    ds = ds.batch(batch_size)

    ds = ds.prefetch(AUTO)

    return ds
    
def build_model(config):
    model = None
    return model

def train(model,config,dataloader):
    return model

def test(model,dataloader):
    return True

def main():

    SEED = 42
    IMG_SIZE = 192
    BATCH_SIZE = 64
    config = {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
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
        batch_size=BATCH_SIZE)
    ds_valid = read_dataset(
        files=files_valid,
        batch_size=BATCH_SIZE)
    ds_test  = read_dataset(
        files=files_test,
        batch_size=BATCH_SIZE)

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