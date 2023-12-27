import gin, logging
import tensorflow as tf 

def read_tfrecords(record):
    # parse the record
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'feature': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_data = tf.io.parse_single_example(record, feature_description)

    # decode and return
    features = tf.io.parse_tensor(parsed_data['feature'], out_type=tf.double)
    labels = tf.io.parse_tensor(parsed_data['label'], out_type=tf.double)
    return (features, labels)


@gin.configurable
def load(name, data_dir):
    if name == "har":
        logging.info(f"Preparing dataset {name}...")
        train_raw = tf.data.TFRecordDataset(data_dir + "train.tfrecords")
        test_raw = tf.data.TFRecordDataset(data_dir + "test.tfrecords")
        val_raw = tf.data.TFRecordDataset(data_dir + "val.tfrecords")

        decoded_train = train_raw.map(read_tfrecords)
        decoded_test = test_raw.map(read_tfrecords)
        decoded_val = val_raw.map(read_tfrecords)

        return prepare(decoded_train, decoded_test, decoded_val, "har")
    

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, shuffle_buffer=32):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(shuffle_buffer)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info


# converting labels 1-12 -> 0-11
def preprocess(features, labels):
    labels = tf.cast(labels, tf.int32)
    labels = tf.subtract(labels, 1)
    return features, labels



