from .lrp.train import *
from .lrp import utils
from . import read_mnist
import os
from matplotlib import pyplot as plt

def cnn_test():
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	# Create placeholders for the network
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	cnn = Network([Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
		NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
		NextLinear(1024), ReLU(),
		NextLinear(10)], X, y_)

	# Get a tensor that calculates deeptaylor explanation for the correct class
	#R_simple = cnn.mixed_lrp(y_, "simple")
	R_ab = cnn.mixed_lrp(y_, ["ab", 1.])
	#R_deeptaylor = cnn.deep_taylor(y_)

	# instanciate network by creating a session
	sess = cnn.create_session()
	
	"""
	try: cnn.load_params("yuhu")
	except: pass
	# train the network for 200 train steps
	for i in range(2000):
		acc, _ = sess.run([cnn.accuracy, cnn.train], feed_dict=cnn.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	# save learned params to dir "yuhu"
	cnn.save_params("yuhu")
	"""
	cnn.load_params("yuhu")
	
	print("Now check out accuracy:")
	acc, = sess.run([cnn.accuracy], feed_dict=cnn.feed_dict(mnist.test.next_batch(200)))
	print("Finally: ", acc)

	# get a batch that we use for testing now
	x, T = mnist.train.next_batch(10)
	feed_dict = cnn.feed_dict([x, T])

	
	# Forwarding tests:
	# test if numpy and tensorflow networks are identical
	cnn.layerwise_tfnp_test(x, T)

	# test again if numpy and tensorflow networks are identical:
	# forward the same batch in tf network
	cnn_y = sess.run(cnn.y, feed_dict=cnn.feed_dict([x, T]))
	# get a networ with numpy backend
	npcnn = cnn.to_numpy()
	# use it
	np_y = npcnn.forward(x)
	print("Max error: ",np.absolute(np_y-cnn_y).max())
	

	# LRP Testing
	# simple lrp with tensorflow
	#cnn.simple_test(feed_dict)
	cnn.ab_test(feed_dict); input()

	y, heatmaps = sess.run([cnn.y, R_ab], feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/ab_lrp.png")
	print("Sum(h) / R = ", np.sum(heatmaps[0]), "/", np.sum((y*T)[0]))

	heatmaps, r = cnn.get_numpy_deeptaylor(x, T)
	utils.visualize(x, utils.heatmap, "cooolcnn/x.png")
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/deeptaylor_np.png")

	"""
	# deeptaylor with tensorflow
	heatmaps = sess.run(R_deeptaylor, feed_dict=feed_dict)
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/deeptaylor_tf.png")

	# get deeptaylor with numpy implementation
	heatmaps, _ = cnn.get_numpy_deeptaylor(x, T)
	utils.visualize(x, utils.heatmap, "cooolcnn/x.png")
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/deeptaylor_np.png")
	"""
	cnn.close_sess()

#mlp_test()
cnn_test()