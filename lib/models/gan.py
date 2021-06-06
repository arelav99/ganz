from lib.metrics import mse, mae, calculate_auc
from lib.models.layers import build_generator_gan, build_discriminator_gan

from tensorflow.keras.losses import binary_crossentropy as bce
from tensorflow import keras
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
tf.get_logger().setLevel('INFO')

class Ganomaly:
    def __init__(self, opts):
        self.opts = opts

        if opts.g_weights_path and opts.d_weights_path:
            self.generator = keras.models.load_model(opts.g_weights_path)
            self.discriminator = keras.models.load_model(opts.d_weights_path)
        else:
            self.generator = build_generator_gan(opts)
            self.discriminator = build_discriminator_gan(opts)

        self.optimizer_d = keras.optimizers.Adam(opts.lr)
        self.optimizer_g = keras.optimizers.Adam(opts.lr)
        # self.optimizer_d = keras.optimizers.SGD(opts.lr)
        # self.optimizer_g = keras.optimizers.SGD(opts.lr)

        self.time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    def fit(self, X):
        for epoch in range(self.opts.epochs):
            np.random.shuffle(X)
            self.train_one_epoch(X, epoch)
        self.save_weights()

    def fit_with_test(self, X, test_data):
        x_test, y_test = test_data
        max_score = 0

        for epoch in range(self.opts.epochs):
            np.random.shuffle(X)
            self.train_one_epoch(X, epoch)

            scores = self.get_anomaly_score(x_test)
            roc_auc = calculate_auc(scores, y_test)
            if roc_auc > max_score:
                max_score = roc_auc
                self.save_weights()

            print(
                f"For epoch {epoch} : \n AUC score - {roc_auc}, best so far - {max_score}")

    def train_one_epoch(self, X, epoch):
        for batch_start in tqdm(range(0, X.shape[0], self.opts.batch_size), position=0, leave=True):
            batch = X[batch_start: batch_start + self.opts.batch_size]
            step = batch_start // self.opts.batch_size
            input_ = tf.convert_to_tensor(batch)
            err_g, err_d, grads_gen, grads_descr = self.calculate_losses_and_gradients(
                input_, step)
            self.update_weights(grads_gen, grads_descr)

        print(
            f"For epoch {epoch} : \n Generator loss - {err_g}, discriminator loss - {err_d}")
        if self.opts.visualize_imgs:
            self.visualize(X[:3])

    def calculate_losses_and_gradients(self, input_, step):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            latent_one, fake, latent_two = self.generator(input_)

            feat_real, pred_real = self.discriminator(input_)
            feat_fake, pred_fake = self.discriminator(fake)

            err_g_adv = mse(feat_real, feat_fake)
            err_g_con = mse(fake, input_)
            err_g_enc = mse(latent_one, latent_two)
            err_g = err_g_con * self.opts.w_con + \
                err_g_adv * self.opts.w_adv + \
                err_g_enc * self.opts.w_enc

            err_d_real = bce(tf.squeeze(pred_real), tf.ones(
                shape=(pred_real.shape[0],), dtype=tf.float32))
            err_d_fake = bce(tf.squeeze(pred_fake), tf.zeros(
                shape=(pred_fake.shape[0],), dtype=tf.float32))
            err_d = (err_d_real + err_d_fake) * 0.5

        grads_gen = tape1.gradient(err_g, self.generator.trainable_weights)
        grads_descr = tape2.gradient(
            err_d, self.discriminator.trainable_weights)

        if step % self.opts.print_steps_freq == 0 and step != 0:
            print(
                f"\n Current step - {step}, generator loss - {err_g}, discriminator loss - {err_d}")
            print(f"Loss between two hidden verctors - {err_g_enc},\
              loss between image and generated image - {err_g_con}, \
              loss between two encoders on true/fake images - {err_g_adv}")

        return err_g, err_d, grads_gen, grads_descr

    def update_weights(self, grads_gen, grads_descr):
        self.optimizer_g.apply_gradients(
            zip(grads_gen, self.generator.trainable_weights))
        self.optimizer_d.apply_gradients(
            zip(grads_descr, self.discriminator.trainable_weights))

    def save_weights(self):
        if not os.path.exists(self.opts.save_weights_path):
            os.makedirs(self.opts.save_weights_path)
        self.generator.save(os.path.join(
            self.opts.save_weights_path, "gen" + self.time_stamp))
        self.discriminator.save(os.path.join(
            self.opts.save_weights_path, "discr" + self.time_stamp))

    def visualize(self, imgs):
        _, fake_imgs, _ = self.generator(imgs)
        for inx in range(imgs.shape[0]):
            _, ax = plt.subplots(1, 2,  figsize=(5, 2))
            ax[0].imshow(tf.squeeze(imgs[inx]) * 127.5 + 127.5)
            ax[1].imshow(tf.cast(tf.squeeze(fake_imgs[inx]) * 127.5 + 127.5, tf.int16))
            plt.show()

    def get_anomaly_score(self, batch):
        input_ = tf.convert_to_tensor(batch)
        latent_one, _, latent_two = self.generator(input_)
        scores = tf.reduce_mean(
            tf.abs(tf.subtract(latent_one, latent_two)), axis=[1, 2, 3])
        return (scores - np.min(scores)) / np.ptp(scores)
