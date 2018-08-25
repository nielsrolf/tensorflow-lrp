
"""
To see stuff on tensorboard, execute this and then
tensorboard --logdir=.logs/summary
All lines important for tensorboard have the comment "# tensorboard"
"""

from lrp import *
from lrp.data import Data
from lrp import read_mnist
import os
from lrp.evaluate_rule import HeatmapEval
from lrp import utils
import os

import tensorflow as tf

#
# Create Data
#
mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)

with namescope("Data"):
    data = Data(X=mnist.train.images, y=mnist.train.labels,
                X_val=mnist.validation.images, y_val=mnist.validation.labels,
                X_test=mnist.test.images, y_test=mnist.test.labels)

samples = {data.X: data.X_test[:16], data.y_: data.y_test[:16]}
y_ = data.y_test[:16]


#
# Interactive Session for debugging
#
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def tensorboard(trainstep): # tensorboard
    # this will be passed to the fit method as perform_action
    summary = sess.run(merged)
    writer.add_summary(summary, trainstep)

#
# Create Graph
#
with namescope("FCN"):
    fcn = Network([Format(), FirstLinear(784), ReLU(), NextLinear(10)], data.X, data.y_, logdir=".logs/fcn")
    fcn.set_session(sess)
    try:
        fcn.load_params("trained_models/example_fcn")
    except:
        print("Train new fcn")
        fcn.fit(data, lambda i, val_acc: max([0] + val_acc) <= 0.98, perform_action=tensorboard)  # tensorboard
        fcn.save_params("trained_models/example_fcn")

    print("FCN deeptaylor ref")
    H_fcn, C_fcn = fcn.layerwise_lrp(data.y_, "deeptaylor", reference=data.X_val, debug_feed_dict=samples)

with namescope("CNN"):
    cnn = Network([Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
                   NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
                   NextLinear(1024), ReLU(),
                   NextLinear(10)],
                  data.X, data.y_, logdir=".logs/cnn")
    cnn.set_session(sess)

    try:
        cnn.load_params("trained_models/example_cnn")
    except:
        print("Train new cnn")
        cnn.fit(data, lambda i, val_acc: max([0] + val_acc) <= 0.98, perform_action=tensorboard)  # tensorboard
        cnn.save_params("trained_models/example_cnn")

    print("CNN deeptaylor")
    H_cnn, C_cnn = cnn.layerwise_lrp(data.y_, "deeptaylor", debug_feed_dict=samples)
    print("CNN deeptaylor ref")
    H_cnn_ref, C_cnn_ref = cnn.layerwise_lrp(data.y_, "deeptaylor", reference=data.X_val, debug_feed_dict=samples)

#
# Prepare stuff for tensorboard
# collect merged summaries after graph is created
merged = tf.summary.merge_all() # tensorboard
writer = tf.summary.FileWriter('.logs/summary', sess.graph) # tensorboard


def total_eval():
    #
    # Eval heatmaps and conservation
    #
    layer = 0

    h, c, y = sess.run([H_cnn, C_cnn, cnn.y], feed_dict=samples)
    h_total = np.sum(h[layer], axis=tuple(range(1, len(h[layer].shape))))
    print("cnn deeptaylor layerwise", c)
    print("total", h_total/np.sum(y*y_, axis=1))

    h, c, y = sess.run([H_cnn_ref, C_cnn_ref, cnn.y], feed_dict=samples)
    h_total = np.sum(h[layer], axis=tuple(range(1, len(h[layer].shape))))
    print("cnn deeptaylor ref layerwise", c)
    print("total", h_total/np.sum(y*y_, axis=1))

    print("------FCN-----")
    h, c, y = sess.run([H_fcn, C_fcn, fcn.y], feed_dict=samples)
    h_total = np.sum(h[layer], axis=tuple(range(1, len(h[layer].shape))))
    print("fcn deeptaylor ref layerwise", c)
    print("total", h_total/np.sum(y*y_, axis=1))

# total_eval()

writer.close() # tensorboard
sess.close()