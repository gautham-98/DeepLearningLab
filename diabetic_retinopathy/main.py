import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from diabetic_retinopathy.utils import utils_params, utils_misc
from models.architectures import vgg_like, cnn
from input_pipeline import tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', False,
                     'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'cnn', 'Choose model to train. Default model cnn')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if tfrecords.make_tfrecords():
        logging.info("Created TFRecords files")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))
    # model
    # if FLAGS.model_name == 'cnn01':
    #     model = cnn01()
    model = cnn()

    if FLAGS.train:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info("Starting model training...")
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    if FLAGS.eval:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info(f"Starting model evaluation...")
        evaluate(model,
                 ds_test,
                 ds_info,
                 run_paths
                 )


if __name__ == "__main__":
    app.run(main)
