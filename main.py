from .lrp.train import *
from .lrp import utils
from . import read_mnist
import os

def mlp_test():
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	mlp = Network([FirstLinear(300), ReLU(), NextLinear(100), ReLU(), NextLinear(10)], X, y_)
	R = mlp.deep_taylor(y_)
	R_simple = mlp.mixed_lrp(y_, "simple")
	R_abdt = mlp.mixed_lrp(y_, methods=[["deep_taylor"], ["ab", 1.], ["ab", 1.], ["ab", 1.], ["ab", 1.]])
	R_ab = mlp.mixed_lrp(y_, methods=[["ab", 2.], ["ab", 2.], ["ab", 2.], ["ab", 2.], ["ab", 2.]])
	R_abb = mlp.mixed_lrp(y_, methods=[["abb", 2.], ["abb", 2.], ["abb", 2.], ["abb", 2.], ["abb", 2.]])
	R_simpleb = mlp.mixed_lrp(y_, "simpleb")
	sess = mlp.create_session()

	"""
	for i in range(2000):
		acc, _ = sess.run([mlp.accuracy, mlp.train], feed_dict=mlp.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	mlp.save_params("yiha")
	"""
	mlp.load_params("yiha")
	acc, _ = sess.run([mlp.accuracy, mlp.train], feed_dict=mlp.feed_dict(mnist.test.next_batch(200)))
	print("Finally: ", acc)

	X, T = mnist.train.next_batch(10); feed_dict=mlp.feed_dict([X, T])

	#mlp.test(X, T)
	print("\n----------------------\n")
	heatmaps, y = sess.run([R, mlp.y], feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps, utils.heatmap, "yiha/deeptaylor_tf.png")
	print("Deeptaylor Conservation: ")
	correct_class_relevance = np.sum(y*T, axis=1)
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps, _ = mlp.get_numpy_deeptaylor(X, T)
	utils.visualize(X, utils.heatmap, "yiha/x.png")
	utils.visualize(heatmaps, utils.heatmap, "yiha/deeptaylor_np.png")
	print("Numpy Deeptaylor Conservation: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps = sess.run(R_abdt, feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps, utils.heatmap, "yiha/ab_deeptaylor.png")
	print("AB Deeptaylor Conservation: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps = sess.run(R_ab, feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps, utils.heatmap, "yiha/ab.png")
	print("AB Conervation: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps_simple = sess.run(R_simple, feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps_simple, utils.heatmap, "yiha/simple_heatmap.png")
	print("Simple Conservation: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps = sess.run(R_simpleb, feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps, utils.heatmap, "yiha/simpleb_heatmap.png")
	print("Simple with flat biases Conservation: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	heatmaps = sess.run(R_abb, feed_dict=mlp.feed_dict([X, T]))
	utils.visualize(heatmaps, utils.heatmap, "yiha/abb_heatmap.png")
	print("AB Conservation with flat biases: ")
	mlp.conservation_check(heatmaps, correct_class_relevance)
	input()

	mlp.close_sess()

def cnn_test():
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	cnn = Network([FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
		NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
		NextLinear(1024), ReLU(),
		NextLinear(10)], X, y_)
	sess = cnn.create_session()
	"""
	for i in range(200):
		acc, _ = sess.run([cnn.accuracy, cnn.train], feed_dict=cnn.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	
	cnn.save_params("yuhu")
	"""
	cnn.load_params("yuhu")
	acc, _ = sess.run([cnn.accuracy, cnn.train], feed_dict=cnn.feed_dict(mnist.test.next_batch(200)))
	print("Finally: ", acc)

	x, T = mnist.train.next_batch(10)
	#cnn.test(x, T)

	npcnn = cnn.to_numpy()
	np_y = npcnn.forward(x)
	cnn_y = sess.run(cnn.y, feed_dict=cnn.feed_dict([x, T]))
	print(np_y-cnn_y)
	input()




	heatmaps, _ = cnn.get_numpy_deeptaylor(x, T)
	utils.visualize(x, utils.heatmap, "cooolcnn/x.png")
	utils.visualize(heatmaps, utils.heatmap, "cooolcnn/correct_class.png")

	E = np.eye(10)
	for c, e in enumerate(E):
		heatmaps, _ = cnn.get_numpy_deeptaylor(x, e)
		utils.visualize(heatmaps, utils.heatmap, "cooolcnn/class_{}.png".format(c))
	nothing, _ = cnn.get_numpy_deeptaylor(x, np.zeros(10))
	utils.visualize(nothing, utils.heatmap, "cooolcnn/nothing.png")

	cnn.close_sess()


cnn_test()