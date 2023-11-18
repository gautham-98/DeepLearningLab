import logging

import gin
import tensorflow as tf
from evaluation.metrics import ConfusionMatrix


@gin.configurable
def evaluate(model, ds_test, ds_info, run_paths):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Restore model to the latest checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train'])).expect_partial()
    logging.info(f"Check point restored from {run_paths['path_ckpts_train']} ")

    confusion_matrix = ConfusionMatrix()
    auc_metric = tf.keras.metrics.AUC(num_thresholds=50)

    # perform predictions on test set
    for images, labels in ds_test:
        predictions = model(images, training=False)
        test_accuracy.update_state(labels, predictions)
        y_pred = tf.argmax(predictions, axis=1)
        # y_preds = tf.expand_dims(y_pred, axis=1)
        y_true = tf.squeeze(labels, axis=-1)
        confusion_matrix.update_state(y_true, y_pred)
        auc_metric.update_state(y_true, y_pred)

    cm_result = confusion_matrix.result()
    auc_result = auc_metric.result()
    ub_accuracy, recall, precision, f1_score, sensitivity, specificity, balanced_accuracy = confusion_matrix.get_related_metrics()
    confusion_matrix.reset_state()
    auc_metric.reset_state()
    sparse_accuracy = test_accuracy.result()

    logging.info(f"\n====Results of Test set evaluation on {model.name} ====")
    logging.info(f"Confusion Matrix: {cm_result.numpy()[0]} {cm_result.numpy()[1]}")
    logging.info("Accuracy(Unbalanced): {:.2f}".format(ub_accuracy * 100))
    logging.info("recall: {:.2f}".format(recall * 100))
    logging.info("precision: {:.2f}".format(precision * 100))
    logging.info("f1_score: {:.2f}".format(f1_score * 100))
    logging.info("sensitivity: {:.2f}".format(sensitivity * 100))
    logging.info("specificity: {:.2f}".format(specificity * 100))
    logging.info("Accuracy(balanced): {:.2f}".format(balanced_accuracy * 100))
    logging.info("Accuracy(Sparse Categorical) {:.2f}".format(sparse_accuracy * 100))
    logging.info("AUC {}".format(auc_result.numpy()))

    logging.info("----Evaluation completed----")
    return