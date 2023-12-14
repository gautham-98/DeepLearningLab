import logging
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import vgg_like, cnn_1
from train import Trainer
from utils import utils_params, utils_misc
from evaluation.eval import evaluate
import os
from train import Trainer
from input_pipeline import tfrecords
from evaluation.eval import evaluate
from absl import app



def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder(','.join(bindings))

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        run.name = run_paths['path_model_id'].split(os.sep)[-1]

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))

        # model
        model = vgg_like()
        #model = cnn_1()
        model.summary()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue

        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info("Starting model evaluation after sweep")
        evaluate(model,
                 ds_test,
                 ds_info,
                 run_paths
                 )


sweep_config = {
    'name': 'diabetic-retinopathy-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [5000, 7000, 7500, 8000, 10000]
        },
        'Trainer.learning_rate': {
            'values': [0.001, 0.003, 0.005, 0.0001, 0.0003]
        },
        'vgg_like.base_filters': {
            'values': [4, 8, 16]
        },
        'vgg_like.n_blocks': {
            'values': [4,5,6,7]
        },
        'vgg_like.dense_units': {
            'values': [16, 32, 64]
        },
        'vgg_like.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=50)
