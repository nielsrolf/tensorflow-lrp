from .lrp.train import *
from .lrp import utils
from . import read_mnist
import os
from matplotlib import pyplot as plt
from .examples import models

def cnn_test(architecture, param_path=None, train_new=False):
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	# Create placeholders for the network
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	
	cnn = Network(architecture, X, y_)

	# Get a tensor that calculates deeptaylor explanation for the correct class
	#R_simple = cnn.mixed_lrp(y_, "simple")
	R_ab = cnn.lrp(y_, ["ab", 2.])
	R_zbab = cnn.lrp(y_, "zbab")
	R_simple = cnn.lrp(y_, "simple")
	R_deeptaylor = cnn.lrp(y_, "deeptaylor")

	# instanciate network by creating a session
	sess = cnn.create_session()
	
	if train_new:
		# train the network for 200 train steps
		for i in range(1000):
			acc, _ = sess.run([cnn.accuracy, cnn.train], feed_dict=cnn.feed_dict(mnist.train.next_batch(50)))
			print(acc)
		cnn.save_params(param_path)
	cnn.load_params(param_path)
	
	print("Now check out accuracy:")
	acc, = sess.run([cnn.accuracy], feed_dict=cnn.feed_dict(mnist.test.next_batch(200)))
	print("Finally: ", acc)

	# get a batch that we use for testing now
	x, T = mnist.train.next_batch(10)
	feed_dict = cnn.feed_dict([x, T])
	
	y, heatmaps = sess.run([cnn.y, R_ab], feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/ab_lrp")
	print("Sum(h) / R = ", np.sum(heatmaps), "/", np.sum((y*T)))

	y, heatmaps = sess.run([cnn.y, R_zbab], feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/zbab_lrp")
	print("Sum(h) / R = ", np.sum(heatmaps), "/", np.sum((y*T)))
	print("lowest:", np.amin(heatmaps))

	y, heatmaps = sess.run([cnn.y, R_simple], feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/simple_lrp")
	print("Sum(h) / R = ", np.sum(heatmaps), "/", np.sum((y*T)))

	# deeptaylor with tensorflow
	heatmaps = sess.run(R_deeptaylor, feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/deeptaylor_tf")
	
	cnn.close_sess()

def conservation_test(architecture, param_path):
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	# Create placeholders for the network
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	nn = Network(architecture, X, y_)
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	
	sess = nn.create_session()
	nn.load_params(param_path)
	acc, = sess.run([nn.accuracy], feed_dict=nn.feed_dict(mnist.test.next_batch(200)))
	print("Check networ with accuracy: ", acc)

	# get some data
	x, T = mnist.train.next_batch(10)
	feed_dict = nn.feed_dict([x, T])

	for methods in ["simple", ["ab", 2.], "zbab", "deeptaylor"]:
		print("\n____________________________________________________________")
		print("Conservation test: ", methods)
		R_layerwise, Conservation_layerwise = nn.layerwise_lrp(y_, methods)
		nn.layerwise_conservation_test(R_layerwise, Conservation_layerwise, feed_dict)

#mlp_test()
cnn_test(models.cnn_5, "cnn_5", False)
#conservation_test(models.cnn_3, "cnn_3")