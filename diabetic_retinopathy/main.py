import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from diabetic_retinopathy.utils import utils_params, utils_misc
from models.architectures import vgg_like, cnn01
from input_pipeline import tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')  # TODO make this to true to train the model
flags.DEFINE_string('model_name', 'cnn01', 'Specify the name of the model to be used to train')


def main(argv):
    print("Hello")

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # create tf-records folder and files if they do not exist yet
    if tfrecords.make_tfrecords():
        logging.info("Created TFRecords files at path specified in gin file")
    else:
        logging.info("TFRecords files already exist. Proceed with the execution")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))

    print("Data is ready, now entering to model part")
    # model
    if FLAGS.model_name == 'cnn01':
        model = cnn01()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)  # TODO make train flag to true for training
        for _ in trainer.train():
            continue
    else:
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)


if __name__ == "__main__":
    app.run(main)
