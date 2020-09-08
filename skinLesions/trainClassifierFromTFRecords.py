# depdencies
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

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

def read_dataset(files,batch_size):
    """
    Read and deserialize the dataset from TFRecord files
    """
    # retrieve raw dataset
    ds  = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    # parse raw dataset
    ds = ds.map(_deserialize_example)
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

    IMG_SIZE = 192
    SEED = 42
    BATCH_SIZE = 32

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

    config = {}
    dataloader = None

    model = build_model(config = config)
    model = train(
        model = model,
        dataloader = dataloader,
        config = config)

    test(model,dataloader)

if __name__ == '__main__':
    main()