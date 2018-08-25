
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

# train a good cnn and a good fcn
mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)

with namescope("Data"):
    data = Data(X=mnist.train.images, y=mnist.train.labels,
                X_val=mnist.validation.images, y_val=mnist.validation.labels,
                X_test=mnist.test.images, y_test=mnist.test.labels)

with namescope("FCN"):
    fcn = Network([Format(), FirstLinear(784), ReLU(), NextLinear(10)], data.X, data.y_, logdir=".logs/fcn")

    H_fcn, C_fcn = fcn.layerwise_lrp(data.y_, "deeptaylor", reference=data.X_val)

with namescope("CNN"):
    cnn = Network([Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
                   NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
                   NextLinear(1024), ReLU(),
                   NextLinear(10)],
                  data.X, data.y_, logdir=".logs/cnn")

    H_cnn, C_cnn = cnn.layerwise_lrp(data.y_, "deeptaylor", reference=data.X_val)


merged = tf.summary.merge_all() # tensorboard


sess = tf.Session()
writer = tf.summary.FileWriter('.logs/summary', sess.graph) # tensorboard


def tensorboard(trainstep): # tensorboard
    # this will be passed to the fit method as perform_action
    summary = sess.run(merged)
    writer.add_summary(summary, trainstep)


fcn.set_session(sess)
cnn.set_session(sess)

sess.run(tf.global_variables_initializer())

try:
    raise Exception("skip")
    fcn.load_params("trained_models/example_fcn")
except:
    print("Train new fcn")
    fcn.fit(data, lambda i, val_acc: max([0] + val_acc) <= 0.98, perform_action=tensorboard) # tensorboard
    fcn.save_params("trained_models/example_fcn")

try:
    raise Exception("skip")
    cnn.load_params("trained_models/example_cnn")
except:
    print("Train new cnn")
    cnn.fit(data, lambda i, val_acc: max([0] + val_acc) <= 0.98, perform_action=tensorboard) # tensorboard
    cnn.save_params("trained_models/example_cnn")

samples = {data.X: data.X_test[:16], data.y_: data.y_test[:16]}
y_ = data.y_test[:16]


layer = 0
h, c, y = sess.run([H_cnn, C_cnn, cnn.y], feed_dict=samples)
h_total = np.sum(h[layer], axis=tuple(range(1, len(h[layer].shape))))
print("deeptaylor ref layerwise", c)
print("total", h_total/np.sum(y*y_, axis=1))

print("------FCN-----")
layer = 0
h, c, y = sess.run([H_fcn, C_fcn, fcn.y], feed_dict=samples)
h_total = np.sum(h[layer], axis=tuple(range(1, len(h[layer].shape))))
print("deeptaylor ref layerwise", c)
print("total", h_total/np.sum(y*y_, axis=1))



writer.close() # tensorboard
sess.close()