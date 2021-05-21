import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from glob import glob


def load_custom_dataset(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("No such directory")
    child = [os.path.join(dataset_path, o) for o in os.listdir(dataset_path)
             if os.path.isdir(os.path.join(dataset_path, o))]

    if not ("train" in "\t".join(child) and "test" in "\t".join(child)):
        raise ValueError("Keep the scructure as in ReadMe")

    train_folder = os.path.join(dataset_path, "train")
    test_folder = os.path.join(dataset_path, "test")
    x_train_true = np.expand_dims(np.array(
        [cv2.imread(img) for img in glob(os.path.join(train_folder, "true/*.jpg"))]), axis=-1)
    y_train = np.zeros(x_train_true.shape[0])

    x_test_true = np.expand_dims(np.array(
        [cv2.imread(img) for img in glob(os.path.join(test_folder, "true/*.jpg"))]), axis=-1)
    x_test_fake = np.expand_dims(np.array(
        [cv2.imread(img) for img in glob(os.path.join(test_folder, "fake/*.jpg"))]), axis=-1)

    y_test = np.concatenate(
        [np.zeros(x_test_true.shape[0]), np.ones(x_test_fake.shape[0])], axis=0)

    return x_train_true, np.vstack([x_test_true, x_test_fake]), y_test


def load_dataset(dataset_name, abnormal_class):
    if dataset_name not in ["cifar", "mnist"]:
        return load_custom_dataset(dataset_name)

    parent_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir))
    if not os.path.exists(os.path.join(parent_dir, "data")):
        os.makedirs(os.path.join(parent_dir, "data"))
    if dataset_name == "cifar":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data(
            os.path.join(parent_dir, "data\\cifar10.npz"))
    elif dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
            os.path.join(parent_dir, "data\\mnist.npz"))

    x_train = 2.*(x_train - np.min(x_train))/np.ptp(x_train)-1
    x_test = 2.*(x_test - np.min(x_test))/np.ptp(x_test)-1

    if dataset_name == "mnist":
        x_train = np.expand_dims([cv2.resize(x_train[idx], (32, 32))
                                  for idx in range(x_train.shape[0])], axis=-1)
        x_test = np.expand_dims([cv2.resize(x_test[idx], (32, 32))
                                 for idx in range(x_test.shape[0])], axis=-1)

    return split_dataset((x_train.astype("float32"), y_train.astype("float32")),
                         (x_test.astype("float32"), y_test.astype("float32")), abnormal_class)


def split_dataset(train_set, test_set, abnormal_class):
    (x_train, y_train), (x_test, y_test) = train_set, test_set
    X_tr = x_train[np.where(y_train != abnormal_class)[0]]
    x_train_fake = x_train[np.where(y_train == abnormal_class)[0]]

    x_test_true = x_test[np.where(y_test != abnormal_class)[0]]
    x_test_fake = x_test[np.where(y_test == abnormal_class)[0]]
    X_tst = np.concatenate([x_test_true, x_test_fake, x_train_fake], axis=0)
    Y_tst = np.concatenate(
        [np.zeros(x_test_true.shape[0]), np.ones(x_test_fake.shape[0]),
         np.ones(x_train_fake.shape[0])], axis=0)
    return X_tr, X_tst, Y_tst
