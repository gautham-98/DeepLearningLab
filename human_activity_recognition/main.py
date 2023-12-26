import gin, logging, sys, os
from input_pipeline import datasets, tfrecords
from absl import app, flags
from utils import utils_params, utils_misc
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', False, 'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'cnn_se', 'Choose model to train. Default model cnn')


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



if __name__ == '__main__':
    app.run(main)

