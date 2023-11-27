import os
import gin
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import logging

from datetime import datetime
from deep_visu.grad_cam import GradCam
from input_pipeline.tfrecords import preprocess_image, convert_to_binary
from input_pipeline.preprocessing import preprocess

@gin.configurable
class DeepVisualize:
    def __init__(self, model, run_paths, data_dir, target_dir, layer_name, image_list_test=None, image_list_train=None):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.run_paths = run_paths
        self.model = model
        self.layer_name = layer_name
        self.image_list_test = image_list_test
        self.image_list_train = image_list_train
        if (image_list_train is None) or (image_list_test is None):
            logging.info("No images specified from either test or train set. gradCAM terminated")
            sys.exit(0)

    def create_dataset(self):
        labels_path = self.data_dir + "labels/"
        images_path = self.data_dir + "images/"

        df_train = pd.read_csv(labels_path + "train.csv", usecols=['Image name', 'Retinopathy grade'])
        df_test = pd.read_csv(labels_path + "test.csv", usecols=['Image name', 'Retinopathy grade'])
        df_train = df_train.sort_values(by='Image name')
        df_test = df_test.sort_values(by='Image name')

        train_idx = [i - 1 for i in self.image_list_train]
        test_idx = [i - 1 for i in self.image_list_test]

        df_train = df_train.iloc[train_idx]
        df_test = df_test.iloc[test_idx]

        file_paths_train = [(images_path + "train/" + filename + ".jpg") for filename in df_train['Image name']]
        file_paths_test = [(images_path + "test/" + filename + ".jpg") for filename in df_test['Image name']]
        images_train = [self.load_and_preprocess(file_path) for file_path in file_paths_train]
        images_test = [self.load_and_preprocess(file_path) for file_path in file_paths_test]

        convert_to_binary(df_train)
        convert_to_binary(df_test)
        labels_train = df_train['Retinopathy grade'].values
        labels_test = df_test['Retinopathy grade'].values

        ds_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
        ds_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
        return ds_train, ds_test

    def load_and_preprocess(self, img_path):
        image = cv2.imread(img_path)
        image = preprocess_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def visualize(self):
        logging.info("===============Starting Deep Visualisation================")
        ds_train, ds_test = self.create_dataset()
        ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logging.info("dataset created from image checkpoints")
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train']))
        logging.info(f"model loaded with checkpoint from {self.run_paths['path_ckpts_train']}")
        # checkpoint.restore(tf.train.latest_checkpoint('/home/RUS_CIP/st184914/dl-lab-23w-team10/experiments/run_2023-11-27T19-00-24-470804/ckpts'))

        output_dir = os.path.join(self.target_dir, 'gradcam_out')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        out_dir = f"out_at_{timestamp}"
        gradcam_out_dir = self.target_dir + "gradcam_out/" + f"{out_dir}/"
        os.makedirs(gradcam_out_dir)

        train_dir = os.path.join(gradcam_out_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        test_dir = os.path.join(gradcam_out_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        logging.info("applying gradCAM")
        gradcam_test = GradCam(self.model, self.layer_name, ds_test)
        gradcam_train = GradCam(self.model, self.layer_name, ds_train)
        gradcam_train.apply_gradcam(self.image_list_train, train_dir)
        gradcam_test.apply_gradcam(self.image_list_test, test_dir)
        logging.info(f"images saved in {gradcam_out_dir}")
