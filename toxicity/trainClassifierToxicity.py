import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string

INPUT_PATH = 'input/'

BATCH_SIZE = 32
EPOCHS = 10
MAX_FEATURES = 10000
MAX_SEQ_LEN = 250
EMBEDDING_DIM = 16
IDENTITY_COLUMNS = [
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€™√²—'

SEED = 42

# Load raw data
df = pd.read_csv(
    f'{INPUT_PATH}/train.csv')
DATASET_SIZE = df.shape[0]

# Generate TF dataset
target = df.pop(TARGET_COLUMN).round().astype('int')
text = df.pop(TEXT_COLUMN)
dataset = tf.data.Dataset.from_tensor_slices(
    (text.values, target.values))


# Split train/valid/test datasets
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
valid_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.batch(
    batch_size=BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.batch(
    batch_size=BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.batch(batch_size=BATCH_SIZE).cache().prefetch(
    buffer_size=AUTOTUNE)


def standardize_text(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_text = tf.strings.regex_replace(
        lowercase, f'[{CHARS_TO_REMOVE}]', ' ')
    return tf.strings.regex_replace(
        stripped_text,
        '[%s]' % re.escape(string.punctuation),
        '')


vectorize_layer = TextVectorization(
    standardize=standardize_text,
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=MAX_SEQ_LEN)

train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label



train_ds = train_ds.map(vectorize_text)
valid_ds = valid_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

