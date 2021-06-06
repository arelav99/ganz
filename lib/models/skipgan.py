from lib.metrics import mse, mae, mse_batch, mae_batch, calculate_auc
from lib.models.layers import build_generator_skipgan, build_discriminator_skipgan


from tensorflow.keras.losses import binary_crossentropy as bce
from tensorflow.keras.layers import Input
from tensorflow.keras import layers as l
from tensorflow import keras
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


class SkipGanomaly:
    def __init__(self, opts):
        self.opts = opts
        assert opts.size % 16 == 0, "Please resize image so that img % 16 == 0"

        if opts.g_weights_path and opts.d_weights_path:
            self.generator = keras.models.load_model(opts.g_weights_path)
            self.discriminator = keras.models.load_model(opts.d_weights_path)
        else:
            self.generator = build_generator_skipgan(opts)
            self.discriminator = build_discriminator_skipgan(opts)

        self.optimizer_g = keras.optimizers.Adam(opts.lr)
        self.optimizer_d = keras.optimizers.Adam(opts.lr)
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
            fake = self.generator(input_)

            feat_real, pred_real = self.discriminator(input_)
            feat_fake, pred_fake = self.discriminator(fake)

            err_g_con = mae(input_, fake)
            err_g_lat = mse(feat_real, feat_fake)
            err_g_adv = bce(tf.squeeze(pred_fake),  tf.ones(pred_fake.shape[0], dtype=np.float32))
            err_g = err_g_con * self.opts.w_con + \
                err_g_lat * self.opts.w_enc + \
                err_g_adv * self.opts.w_adv

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
            print(f"BCE between prediction on true and fake labels - {err_g_lat},\
              loss between image and generated image - {err_g_con}, \
              loss between two encoders on true/fake images - {err_g_lat}")

        return err_g, err_d, grads_gen, grads_descr

    def get_anomaly_score(self, X):
        scores = []
        for inx in range(0, X.shape[0], 1024):
            if inx != X.shape[0] - X.shape[0] % 1024:
                batch = X[inx: inx+1024]
            else:
                batch = X[inx: inx+X.shape[0] % 1024]

            input = tf.convert_to_tensor(batch)
            fake = self.generator(input)
            feat_real, _ = self.discriminator(input)
            feat_fake, _ = self.discriminator(fake)

            err_g_con_btch = mse_batch(input, fake)
            err_g_lat_btch = mse_batch(feat_real, feat_fake)
            scores.append(.9 * tf.cast(err_g_con_btch,
                          tf.float32) + .1 * err_g_lat_btch)
        scores = list(chain.from_iterable(scores))
        return (scores - np.min(scores)) / np.ptp(scores)

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
        fake_imgs = self.generator(imgs)
        for inx in range(imgs.shape[0]):
            _, ax = plt.subplots(1, 2,  figsize=(5, 2))
            ax[0].imshow(tf.squeeze(imgs[inx]) * 127.5 + 127.5)
            ax[1].imshow(tf.squeeze(fake_imgs[inx]) * 127.5 + 127.5)
            plt.show()
