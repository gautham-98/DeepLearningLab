import logging

import gin
import tensorflow as tf
import numpy as np
from utils import utils_params, utils_misc
from models.architectures import cnn_1, transfer_model, vgg_like, cnn_se
from input_pipeline import datasets
from evaluation.metrics import ConfusionMatrix
from train import Trainer
from input_pipeline import tfrecords
from evaluation.eval import evaluate
from absl import app


def ensemble(models, ds_test, ds_info, ensemble_type="voting"):
    cm= ConfusionMatrix()
    if ensemble_type == "voting":
        for images, labels in ds_test:
            preds = [model(images, training=False) for model in models]
            y_labels = [tf.argmax(pred, axis=-1) for pred in preds]
            labels_sum = tf.reduce_sum(tf.convert_to_tensor(y_labels), axis=0)
            boundary = len(models) // 2
            voted_label = tf.where(labels_sum > boundary, 1, 0)
            cm.update_state(labels, voted_label)

    if ensemble_type == "avg":
        for images, labels in ds_test:
            predictions = [model(images, training=False) for model in models]
            predictions = tf.squeeze(tf.reduce_mean(tf.convert_to_tensor(predictions), axis=0))
            y_pred = tf.argmax(predictions, axis=1)
            cm.update_state(labels, y_pred)



    cm_result = cm.result()
    ub_accuracy, recall, precision, f1_score, sensitivity, specificity, balanced_accuracy = cm.get_related_metrics()

    logging.info(f"\n====Results of Ensemble Evaluation")
    logging.info(f"Confusion Matrix: {cm_result.numpy()[0]} {cm_result.numpy()[1]}")
    logging.info("Accuracy(balanced): {:.2f}".format(balanced_accuracy * 100))
    logging.info("Accuracy(Unbalanced): {:.2f}".format(ub_accuracy * 100))
    logging.info("sensitivity: {:.2f}".format(sensitivity * 100))
    logging.info("specificity: {:.2f}".format(specificity * 100))
    logging.info("recall: {:.2f}".format(recall * 100))
    logging.info("precision: {:.2f}".format(precision * 100))
    logging.info("f1_score: {:.2f}".format(f1_score * 100))

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder("ensemble")

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))

    ckpt_path_1 = "/home/RUS_CIP/st184584/dl-lab-23w-team10/experiments/run_2023-12-14T11-41-13-221007/ckpts" # dense net
    ckpt_path_2 = "/home/RUS_CIP/st184584/dl-lab-23w-team10/experiments/run_2023-12-11T23-04-28-948947/ckpts" # cnn_1
    ckpt_path_3 = "/home/RUS_CIP/st184584/dl-lab-23w-team10/experiments/run_2023-12-14T15-05-59-469451/ckpts" # cnn se

    model1 = transfer_model(base_model_name="DenseNet121")
    ckpt1 = tf.train.Checkpoint(model=model1)
    ckpt1.restore(tf.train.latest_checkpoint(ckpt_path_1))


    model2 = cnn_1()
    ckpt2 = tf.train.Checkpoint(model=model2)
    ckpt2.restore(tf.train.latest_checkpoint(ckpt_path_2))

    model3 = cnn_se()
    ckpt3 = tf.train.Checkpoint(model=model3)
    ckpt3.restore(tf.train.latest_checkpoint(ckpt_path_3))

    
    models = [model1, model2, model3]
    ensemble(models, ds_test, ds_info, ensemble_type="voting")




if __name__ == "__main__":
     app.run(main)

    


    
