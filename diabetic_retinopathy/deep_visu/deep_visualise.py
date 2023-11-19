import os
import gin
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import logging

from diabetic_retinopathy.deep_visu.grad_cam import GradCam

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

        self.convert_to_binary(df_train)
        self.convert_to_binary(df_test)
        labels_train = df_train['Retinopathy grade'].values
        labels_test = df_test['Retinopathy grade'].values

        ds_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
        ds_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
        return ds_train, ds_test

    def load_and_preprocess(self, img_path):
        image = cv2.imread(img_path)
        image = self.preprocess_image(image)
        return image

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        pos = np.nonzero(threshold)
        right_boundary = pos[1].max()
        left_boundary = pos[1].min()
        image = image[:, left_boundary:right_boundary]
        # computations to obtain square image (padding at desired positions)
        upper_diff = (image.shape[1] - image.shape[0]) // 2
        lower_diff = image.shape[1] - image.shape[0] - upper_diff
        image = cv2.copyMakeBorder(image, upper_diff, lower_diff, 0, 0, cv2.BORDER_CONSTANT)
        return cv2.resize(image, (256, 256))

    def convert_to_binary(self, df):
        df.loc[df['Retinopathy grade'] < 2, "Retinopathy grade"] = 0
        df.loc[df['Retinopathy grade'] > 1, "Retinopathy grade"] = 1
        return df

    def visualize(self):
        ds_train, ds_test = self.create_dataset()
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train']))

        output_dir = os.path.join(self.target_dir, 'gradcam_out')
        # Create "gradcam_out" directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        gradcam_out_dir = self.target_dir + "/gradcam_out/"
        train_dir = os.path.join(gradcam_out_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        test_dir = os.path.join(gradcam_out_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        logging.info("===============applying gradCAM================")
        gradcam_test = GradCam(self.model, self.layer_name, ds_test)
        gradcam_train = GradCam(self.model, self.layer_name, ds_train)
        gradcam_train.apply_gradcam(self.image_list_train, train_dir)
        gradcam_test.apply_gradcam(self.image_list_test, test_dir)
        logging.info("=================gradCAM ended=================")
