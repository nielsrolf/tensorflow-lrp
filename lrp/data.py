import numpy as np
import tensorflow as tf
import random
"""
Data interface used by evaluate rule
"""
class Data():
    def __init__(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None, batch_size=50, img_shape=None):
        self.batch_size = batch_size

        ids = np.random.permutation(len(X))
        if X_test is None:
            X_test = X[ids[::5]]
            y_test = y[ids[::5]]
            ids = [ids[i] for i in range(len(ids)) if i%5 != 0]
        if X_val is None:
            X_val = X[ids[::5]]
            y_val = y[ids[::5]]
            ids = [ids[i] for i in range(len(ids)) if i%5 != 0]
        self.X_test, self.y_test = X_test, y_test
        self.X_val, self.y_val = X_val, y_val
        self.X_train, self.y_train = X[ids], y[ids]

        self.X = tf.placeholder(dtype=tf.float32, shape=[None]+list(self.X_train.shape[1:]))
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None]+list(self.y_train.shape[1:]))

        self.epochs = 0
        self.used_this_epoch = set()
        self.is_new_epoch = True
        self.train_ids = set(range(len(self.X_train)))

        if img_shape is not None:
            self.img_shape = img_shape
        elif len(X[0].shape) > 1:
            self.img_shape = self.X_train[0].shape

    def _new_epoch(self):
        self.used_this_epoch = set()
        self.epochs += 1
        self.is_new_epoch = True

    def next_batch(self, batch_size=None):
        """
        :param batch_size: number of rows
        :return feed dict
        """
        if len(self.used_this_epoch) == len(self.train_ids):
            self._new_epoch()
        else:
            self.is_new_epoch = False
        batch_size = min(batch_size, len(self.train_ids) - len(self.used_this_epoch))
        ids = random.sample(self.train_ids-self.used_this_epoch, batch_size)
        self.used_this_epoch = self.used_this_epoch.union(set(ids))
        return {self.X: self.X_train[ids], self.y_: self.y_train[ids]}

    def validation_batch(self):
        return {self.X: self.X_val, self.y_: self.y_val}

    def test_batch(self):
        return {self.X: self.X_test, self.y_: self.y_test}



