from lrp.train import *
from lrp import read_mnist
import os
from matplotlib import pyplot as plt


def visualize_perceptron_filter(w, export_dir):
	"""
	For a Linear one layer perceptron
	"""
	tf.gfile.MakeDirs(export_dir)
	w_flat = np.reshape(w, np.prod(w.shape))

	min_w = w_flat.min()
	max_w = w_flat.max()
	for filter_id in range(shape(w)[-1]):
		w_ = np.reshape(w[:,filter_id], [28, 28])

		img = np.pad(w_, 1, 'constant', constant_values=min_w)
		img = np.pad(img, 1, 'constant', constant_values=max_w) 
		plt.imshow((img), cmap="gray")
		plt.savefig("{}/{}.png".format(export_dir, filter_id))
		plt.clf()

def visualize_filter(w, export_dir):
	# assume w is weight tensor of cn layer with input channel = 1
	tf.gfile.MakeDirs(export_dir)
	w_flat = np.reshape(w, np.prod(w.shape))

	min_w = w_flat.min()
	max_w = w_flat.max()
	for filter_id in range(shape(w)[-1]):

		img = np.pad(w[:,:,0,filter_id], 1, 'constant', constant_values=min_w)
		img = np.pad(img, 1, 'constant', constant_values=max_w) 
		plt.imshow((img), cmap="gray")
		plt.savefig("{}/{}.png".format(export_dir, filter_id))
		plt.clf()

def plot_cnn_filters():
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	cnn = Network([Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
		NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
		NextLinear(1024), ReLU(),
		NextLinear(10)], X, y_)
	sess = cnn.create_session()
	cnn.load_params("yuhu")
	w = sess.run(cnn.layers[1].weights, feed_dict={})
	visualize_filter(w, "yuhu")


def bigfilter_cnn():
	"""
	One layer linear perceptron, implemented as cnn
	"""
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	cnn = Network([Format(), FirstConvolution([28, 28, 1, 10]), Flatten()], X, y_)
	sess = cnn.create_session()
	for i in range(5000):
		_, acc = sess.run([cnn.train, cnn.accuracy], feed_dict=cnn.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	cnn.save_params("yuhu")

	x, T = mnist.test.next_batch(10)
	"""
	heatmaps, _ = cnn.get_numpy_deeptaylor(x, T)
	utils.visualize(x, utils.heatmap, "yuhu/x.png")
	utils.visualize(heatmaps, utils.heatmap, "yuhu/deeptaylor_np.png")
	"""

	w = sess.run(cnn.layers[1].weights, feed_dict={})
	visualize_filter(w)

def perceptron_filter():
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	discriminant_filter = tf.constant([-1.]*10, dtype=tf.float32) + 2*y_

	# specify a network architecture
	cnn = Network([Format(), FirstLinear(10)], X, y_)
	sess = cnn.create_session()
	for i in range(5000):
		_, acc = sess.run([cnn.train, cnn.accuracy], feed_dict=cnn.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	cnn.save_params("yuhu")

	x, T = mnist.test.next_batch(10)
	"""
	heatmaps, _ = cnn.get_numpy_deeptaylor(x, T)
	utils.visualize(x, utils.heatmap, "yuhu/x.png")
	utils.visualize(heatmaps, utils.heatmap, "yuhu/deeptaylor_np.png")
	"""

	w = sess.run(cnn.layers[1].weights, feed_dict={})
	visualize_perceptron_filter(w, "perceptron_filter")