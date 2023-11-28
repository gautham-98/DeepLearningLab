import gin
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import os
import cv2


@gin.configurable
class GradCam:
    def __init__(self, model, layer_name, dataset, class_idx=None):
        self.model = model
        self.layer_name = layer_name
        self.dataset = dataset
        self.class_idx = class_idx
        self.grad_model = tf.keras.models.Model(self.model.inputs,
                                                [self.model.get_layer(name=self.layer_name).output, self.model.output])
    
    @tf.function
    def get_average_grads(self, image):
        with tf.GradientTape() as tape:
            layer_activations, preds = self.grad_model(image)
            if self.class_idx is None:
                self.class_idx = tf.argmax(preds[0])
            class_channel = preds[:, self.class_idx]
            grads = tape.gradient(class_channel, layer_activations)

        average_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        return average_grads, layer_activations


    def get_heatmap(self, image):
        average_grads, layer_activations = self.get_average_grads(image)
        average_grads = tf.expand_dims(average_grads, axis=-1)
        heatmap = layer_activations @ average_grads
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()


    def get_jet_heatmap(self, heatmap, image):
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        jet_heatmap = jet_heatmap/255.0
        return jet_heatmap


    def apply_gradcam(self, image_list, save_dir, alpha=0.5):
        dataset = self.dataset
        for i, (image, label) in enumerate(dataset):
            img_idx = image_list[i]
            image = tf.expand_dims(image, axis=0)
            heatmap = self.get_heatmap(image)
            image = tf.squeeze(image, axis=0)
            jet_heatmap = self.get_jet_heatmap(heatmap, image)
            superimposed_img = jet_heatmap * alpha + image
            # superimposed_img = image
            save_path = os.path.join(save_dir, f"img_{img_idx}_label-{label}.jpg")
            tf.keras.preprocessing.image.save_img(save_path, superimposed_img)
