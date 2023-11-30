import gin
import logging
from absl import app, flags
import wandb

from deep_visu.deep_visualise import DeepVisualize
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like, cnn01, res_cnn, transfer_model
from input_pipeline import tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', True, 'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'cnn01', 'Choose model to train. Default model cnn')
flags.DEFINE_boolean('deep_visu', True, 'perform deep visualization with grad_cam')

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if tfrecords.make_tfrecords():
        logging.info("Created TFRecords files")

    # setup wandb
    wandb.init(project='diabetic-retinopathy', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))
    # model
    model = res_cnn()

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
                 )

    if FLAGS.deep_visu:
        deep_visualize = DeepVisualize(model, run_paths, data_dir=gin.query_parameter('make_tfrecords.data_dir'))
        deep_visualize.visualize()


if __name__ == "__main__":
    app.run(main)
