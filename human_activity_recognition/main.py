import gin, logging, sys, os
from input_pipeline import datasets, tfrecords
from absl import app, flags
from utils import utils_params, utils_misc
import warnings
import tensorflow as tf 

from models.architectures import model1_LSTM
from train import Trainer

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', False, 'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'model1_LSTM', 'Choose model to train. Default model cnn')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str()) 

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    if tfrecords.make_tfrecords():
        logging.info("TF Records Created")

    # load the dataset
    ds_train, ds_val, ds_test, ds_info = datasets.load( name="har", data_dir=gin.query_parameter('make_tfrecords.target_dir'))
    logging.info(f"[DATASET loaded!] {ds_info}")

    # model
    if FLAGS.model_name == 'model1_LSTM':
        model = model1_LSTM()

    if FLAGS.train:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info("Starting model training...")
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue


if __name__ == '__main__':
    app.run(main)

