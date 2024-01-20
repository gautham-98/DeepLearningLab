import gin, logging, sys, os
from input_pipeline import datasets, tfrecords, tfrecords_realworldhar
from absl import app, flags
from utils import utils_params, utils_misc
import warnings
import tensorflow as tf 
import wandb
import numpy as np


from models.architectures import model1_LSTM, model_bidirectional_LSTM, model1_GRU, model1D_Conv
from train import Trainer
from evaluation.eval import evaluate

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', True, 'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'model1_LSTM', 'Choose model to train. Default model model1_LSTM')
flags.DEFINE_boolean('hapt', True, 'hapt dataset' ) # UCI HAR dataset
flags.DEFINE_boolean('har', False, 'har dataset' )  # real world har dataset
flags.DEFINE_boolean('createTFliteModel', False, 'create TFlite model')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str()) 

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    if FLAGS.hapt:
        if tfrecords.make_tfrecords():
            logging.info("TF Records HAPT Created")
        data_dir = gin.query_parameter('make_tfrecords.target_dir')
        name="hapt"
        n_classes =12
        window_length = gin.query_parameter('make_tfrecords.window_length')


    # body_parts = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
    if FLAGS.har:
        if tfrecords_realworldhar.make_tfrecords_rwhar(bodypart="chest"):  # make sure that target directory in config.gin name matches the bodypart parameter
            logging.info("TF records Real World HAR 2016 created")
        data_dir=gin.query_parameter('make_tfrecords_rwhar.target_dir')
        name="har"
        n_classes = 8
        window_length = gin.query_parameter('make_tfrecords_rwhar.window_length')

        # setup wandb
    wandb.init(project='diabetic-retinopathy', name=run_paths['path_model_id'],
            config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    
    # load the dataset
    ds_train, ds_val, ds_test, ds_info, class_weights = datasets.load(name=name, data_dir=data_dir)
    logging.info(f"[DATASET loaded!] {ds_info}")

    
    
    # model
    if FLAGS.model_name == 'model1_LSTM':
        model = model1_LSTM(window_length=window_length, n_classes=n_classes)
    elif FLAGS.model_name == 'model_bidirectional_LSTM':
        model = model_bidirectional_LSTM(window_length=window_length, n_classes=n_classes)
    elif FLAGS.model_name == 'model1_GRU':
        model = model1_GRU(window_length=window_length, n_classes=n_classes)
    elif FLAGS.model_name == 'model1D_Conv':
        model = model1D_Conv(window_length=window_length, n_classes=n_classes)

    

    if FLAGS.train:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info("Starting model training...")
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, ds_info, class_weights, run_paths)
        for _ in trainer.train():
            continue
    if FLAGS.eval:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info(f"Starting model evaluation...")
        evaluate(model, ds_test, ds_info)

    if FLAGS.createTFliteModel:
        run_model = tf.function(lambda x: model(x))
        batch_size = 1
        input_size = 6
        concrete_func = run_model.get_concrete_function(tf.TensorSpec([batch_size, window_length, input_size], model.inputs[0].dtype))

        # model directory.
        model_dir = "./TFlite_model"
        model.save(model_dir, save_format="tf", signatures=concrete_func)

        converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
        tflite_model = converter.convert()
        with open(model_dir+"/model.tflite", "wb") as f:
            f.write(tflite_model)



if __name__ == '__main__':
    app.run(main)

