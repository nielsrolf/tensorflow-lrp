from . import modules
from . import utils
import numpy as np 
import tensorflow as tf

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape): return tf.Variable(tf.constant(0.1, shape=shape))

def shape(T):
	if isinstance(T, tf.Tensor): return [-1 if d is None else d for d in T.get_shape().as_list()]
	return [-1 if d is None else d for d in T.shape]

def expand_dims(A, axes=[]):
	for axis in axes:
		A = tf.expand_dims(A, axis=axis)
	return A

# -------------------------
# Network
# -------------------------
class Network():
	def __init__(self, layers, input_tensor, y_):
		self.format_layer = layers[0]
		self.layers = layers[1:]
		self.input_tensor = input_tensor
		self.y_ = y_
		self.session = None

		# forward through every layer
		activation = input_tensor

		print("Network layer mappings:", activation.shape)
		for layer in layers:
			activation = layer.forward(activation)
			print("->", activation.shape)

		self.y = activation

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1)), tf.float32))
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y)
		self.train = tf.train.AdamOptimizer().minimize(self.loss)
		self.saver = tf.train.Saver()

	def lrp(self, class_filter, methods = "simple"):
		"""
		Gets:
			class_filter: relevance for which class? - as one hot vector; for correct class use ground truth placeholder
			Methods: which method to use? Same meaning as in layerwise_lrp
		Returns:
			R: input layers relevance tensor
		"""
		Rs, _= self.layerwise_lrp(class_filter, methods)
		return Rs[0]

	def layerwise_lrp(self, class_filter, methods = "simple"): # class_filter: tf.constant one hot-vector
		"""
		Gets:
			class_filter: relevance for which class? - as one hot vector; for correct class use ground truth placeholder
			Methods: which method to use?
					If the same method should be used for every layer, then the string can be passed: ("simple" / "ab")
					If one of the following standard method combinations shall be used, also pass the string code:
						"zbab" -> zb - rule for the first layer, after that ab-rule with alpha=2.
						"wwab" -> ww - rule for the first layer, after that ab-rule with alpha=2.
					If this is the case, but the methods needs an additional numeric parameter, then it can be passed like ["methodstr", <param>]
					If the methods shall be specified for each layer, then a list has to be passed, where each element is a list like ["methodstr"(, <param>)]
		Returns:
			R_layerwise: list of relevance tensors, one for each layer, so that R[0] is in input-space and R[-1] is in readout-layer space
		"""

		R = tf.multiply(self.y, class_filter)
		print("\nLayerwise Relevance Propagation <3")
		if methods=="zbab":
			methods = [["zb"]] + [["ab", 2.]]*(len(self.layers)-1)
		elif methods=="wwab":
			methods = [["ww"]] + [["ab", 2.]]*(len(self.layers)-1)
		elif isinstance(methods, str):
			methods = [[methods]] * len(self.layers)
		elif isinstance(methods[1], (float, int)):
			methods = [methods] * len(self.layers)

		R_layerwise = [R]
		Conservation_layerwise = [None]

		for l, m in zip(self.layers[::-1], methods[::-1]):
			backward_mapping = str(shape(R)) + " -> "
			if m[0] == "deep_taylor" or  m[0] == "deeptaylor":
				R, C = l.deep_taylor(R)
			elif m[0] == "simple":
				R, C = l.simple_lrp(R)
			elif m[0] == "ab" or m[0] == "alphabeta":
				if len(m)>1:
					R, C = l.alphabeta_lrp(R, m[1])
				else:
					R, C = l.alphabeta_lrp(R)
			elif m[0] == "simpleb":
				R, C = l.simple_lrp(R)
			elif m[0] == "abb" or m[0] == "alphabeta":
				R, C = l.alphabeta_lrp(R, m[1])
			elif m[0] == "zb":
				R, C = l.zB_lrp(R)
			elif  m[0] == "ww":
				R, C = l.ww_lrp(R)
			else:
				raise Exception("Unknown LRP method: {}".format(m[0]))
			backward_mapping += str(shape(R))
			print(type(l), ": ", m, ":", backward_mapping)
			R_layerwise = [R] + R_layerwise
			Conservation_layerwise = [C] + Conservation_layerwise
		return R_layerwise, Conservation_layerwise

	def layerwise_conservation_test(self, R_layerwise, Conservation_layerwise, feed_dict):
		if self.sess is None: raise Exception("Network.layerwise_conservation_test must be called while Network.session is set")
		print("\n")
		r_in = R_layerwise[-1]
		for i_, (l, C, R) in enumerate(zip(self.layers[::-1], Conservation_layerwise[:-1][::-1], R_layerwise[:-1][::-1])):
			i = len(self.layers)-1 - i_
			c, input_array, output_array, r = self.sess.run([C, l.input_tensor, l.output_tensor, R], feed_dict=feed_dict)
			print("Layer {}: {}: Conservation: {}".format(i, type(l), c))
			print("		Forward:", shape(input_array), " -> ", shape(output_array))
			print("		Backward", shape(r), " <- ", shape(r_in))
			print("R_out = ", np.sum(r))

	def layerwise_conservation_test_(self, R_layerwise, Conservation_layerwise, feed_dict):
		if self.sess is None: raise Exception("Network.layerwise_conservation_test must be called while Network.session is set")
		r_in = self.layers[-1]
		conservation_layerwise = self.sess.run(Conservation_layerwise, feed_dict=feed_dict)
		for i, (l, c) in enumerate(zip(self.layers+["filtered forwarded readout layer = Relevance input layer"], conservation_layerwise)):
			print("Layer {}: {}: Conservation: {}".format(i, type(l), c))

	def save_params(self, export_dir):
		tf.gfile.MakeDirs(export_dir)
		save_path = self.saver.save(self.sess, "{}/model.ckpt".format(export_dir))
		print("Saved model to ", save_path)

	def load_params(self, import_dir):
		self.saver.restore(self.sess, "{}/model.ckpt".format(import_dir))

	def close_sess(self):
		try: self.sess.close()
		except: pass

	def create_session(self):
		self.close_sess()
		self.set_session(tf.Session())
		self.sess.run(tf.global_variables_initializer())
		return self.sess

	def set_session(self, sess):
		self.sess = sess
		for layer in self.layers:
			layer.set_session(sess)

	def to_numpy(self):	
		# extract Parameters, and return a modules.network
		lrp_layers = []
		for layer in [self.format_layer] + self.layers:
			lrp_layers.append(layer.to_numpy())
		return modules.Network(lrp_layers)

	def get_numpy_deeptaylor(self, X, dim): # dim: one hot vector; heatmaps for correct class: dim = y_
		# X: either np.array or tf.Tensor
		numpy_network = self.to_numpy()
		y = numpy_network.forward(X)
		return numpy_network.relprop(y*dim), y*dim

	def conservation_check(self, heatmaps, R):
		# input as numpy.array
		num_samples = heatmaps.shape[0]
		h = np.reshape(heatmaps, [num_samples, np.prod(heatmaps.shape[1:])])
		hR = np.sum(h, axis=1)
		R = np.reshape(R, hR.shape)
		err = np.absolute(hR-R)
		print("Relevance - Sum(heatmap) = ", err)

	def ab_test(self, feed_dict):
		ab_errors = self.sess.run([l.ab_forward_error for l in self.layers if isinstance(l, Convolution)], feed_dict=feed_dict)
		for error in ab_errors:
			print("AB forward error: ", np.mean(np.absolute(error))); input()

	def layerwise_tfnp_test(self, X, T):
		np_nn = self.to_numpy()
		cnn_layer_tensors = [layer.output_tensor for layer in self.layers]
		cnn_layer_activations = self.sess.run(cnn_layer_tensors, feed_dict=self.feed_dict([X, T]))
		a = self.format_layer.forward(X)
		for l, a_ in zip(np_nn.layers, cnn_layer_activations):
			a = l.forward(a)
			np.testing.assert_allclose(a, a_, atol=1e-5)
		print("All np/tf layers do the same :) ")
		
	def feed_dict(self, batch):
		return {self.input_tensor: batch[0], self.y_: batch[1]}

# -------------------------
# Abstract Layer
# -------------------------
class Layer():
	def __init__(self):
		self.sess = None

	def set_session(self, sess):
		self.sess = sess

	def to_numpy(self):		# For layers with weights and biases (Linear, Convolution); Pooling and ReLU override this
		if self.sess is None:
			raise Exception("Trying to extract variables without active session")
		else:
			W, B = self.sess.run([self.weights, self.biases], feed_dict={})
			return self.lrp_module(W, B)

# -------------------------
# Format Layer
# -------------------------
class Format(Layer):
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		self.output_tensor = tf.add(input_tensor*(utils.highest-utils.lowest), utils.lowest)
		return self.output_tensor

	def to_numpy(self):
		return modules.Format()

class Flatten(Layer):
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		self.output_tensor = tf.reshape(input_tensor, [-1, np.prod(shape(input_tensor)[1:])])
		return self.output_tensor
	def lrp(self, R): return R, tf.constant(1.)
	def deep_taylor(self, R): return self.lrp(R)
	def simple_lrp(self, R): return self.lrp(R)
	def alphabeta_lrp(self, R, alpha=1.): return self.lrp(R)

# -------------------------
# Activation Layers
# -------------------------
class Activation(Layer):
	def deep_taylor(self, R): return R, tf.constant(1.)
	def simple_lrp(self, R): return R, tf.constant(1.)
	def alphabeta_lrp(self, R, alpha=1.): return R, tf.constant(1.)

class ReLU(Activation):
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		self.output_tensor = tf.nn.relu(input_tensor)
		return self.output_tensor

	def to_numpy(self):
		return modules.ReLU()

class Tanh(Activation):
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		self.output_tensor = tf.nn.tanh(input_tensor)
		return self.output_tensor

class Abs(Activation):
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		self.output_tensor = tf.abs(input_tensor)
		return self.output_tensor

	def to_numpy(self):
		return modules.Abs()

# -------------------------
# Abstract Layer with weights
# -------------------------
class AbstractLayerWithWeights(Layer):
	"""
	Handles forwarding and _simple_lrp based stuff. Child classes must set the following in __init___
	- self.get_w_shape()
	- self.bias_shape
	- self.linear_operation(input_tensor, weights)
	"""
	def forward(self, input_tensor):
		self.input_tensor = self.input_reshape(input_tensor)
		self.weights = weight_variable(self.get_w_shape(self.input_tensor))
		self.biases = bias_variable(self.bias_shape)
		self.activators = self.linear_operation(self.input_tensor, self.weights)
		self.output_tensor = self.activators + self.biases
		return self.output_tensor

	def _simple_lrp(self, R, weights, input_tensor):
		activators = self.linear_operation(input_tensor, weights)
		R = tf.reshape(R, shape(activators))
		R_per_act = tf.divide(R, activators)
		R_per_in_act = tf.gradients(activators, input_tensor, grad_ys=R_per_act)[0]
		R_out = tf.multiply(R_per_in_act, input_tensor)
		return R_out#R_per_in_act#R_out

	def simple_lrp(self, R):
		R_out = self._simple_lrp(R, self.weights, self.input_tensor)
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def alphabeta_lrp(self, R, alpha=1.):
		# WARNING works only if input_tensor is positive
		input_tensor = self.input_tensor + 1e-9
		w_plus = tf.nn.relu(self.weights) +1e-9
		w_minus = tf.nn.relu(-self.weights) +1e-9
		R_out_plus = self._simple_lrp(R, w_plus, input_tensor)
		R_out_minus = self._simple_lrp(R, w_minus, input_tensor)
		R_out = alpha* R_out_plus + (1.-alpha)*R_out_minus
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def alphabeta_lrp_(self, R, alpha=1.):
		# Split activators into + and -
		input_plus = tf.nn.relu(self.input_tensor) +1e-9
		input_minus = tf.nn.relu(-self.input_tensor) +1e-9
		w_plus = tf.nn.relu(self.weights) +1e-9
		w_minus = tf.nn.relu(-self.weights) +1e-9

		a_plus1 = self.linear_operation(input_plus, w_plus)
		a_plus2 = self.linear_operation(input_minus, w_minus)
		a_minus1 = self.linear_operation(input_plus, w_minus)
		a_minus2 = self.linear_operation(input_minus, w_plus)
		# the following can be used to check if activator decomposition holds:
		#self.ab_forward_error = tf.divide(self.activators - (a_plus1+a_plus2 - (a_minus1+a_minus2)), self.activators)

		R_plus1 = tf.multiply(R, tf.divide(a_plus1, a_plus1+a_plus2))
		R_plus2 = tf.multiply(R, tf.divide(a_plus2, a_plus1+a_plus2))
		R_minus1 = tf.multiply(R, tf.divide(a_minus1, a_minus1+a_minus2))
		R_minus2 = tf.multiply(R, tf.divide(a_minus2, a_minus1+a_minus2))

		R_out_plus = self._simple_lrp(R_plus1, w_plus, input_plus) + self._simple_lrp(R_plus2, w_minus, input_minus)
		R_out_minus = self._simple_lrp(R_minus1, w_minus, input_plus) + self._simple_lrp(R_minus2, w_plus, input_minus)

		R_out = alpha* R_out_plus + (1.-alpha)*R_out_minus
		#R_out = R_out_minus

		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

# -------------------------
# Fully-connected layer
# -------------------------
class Linear(AbstractLayerWithWeights):
	def __init__(self, num_neurons):
		super().__init__()
		self.linear_operation = tf.matmul
		self.output_dims = num_neurons

		def input_reshape(input_tensor):
			input_shape = input_tensor.get_shape().as_list()
			if len(input_shape) == 4:
				input_shape = [-1, np.prod(input_shape[1:])]
				input_tensor = tf.reshape(input_tensor, input_shape)
			return input_tensor	
		self.input_reshape=input_reshape
		self.bias_shape = [self.output_dims]

		def get_w_shape(input_tensor):
			print("get w shape: input:", input_tensor)
			return  [shape(input_tensor)[-1], num_neurons]
		self.get_w_shape = get_w_shape#lambda input_tensor : [shape(input_tensor)[-1], num_neurons]

	def zB_lrp(self, R):
		self.R_in = R
		W,V,U = self.weights, tf.maximum(0.,self.weights), tf.minimum(0.,self.weights)
		X,L,H = self.input_tensor, self.input_tensor*0 + utils.lowest, self.input_tensor*0 + utils.highest

		Z = tf.matmul(X,W)-tf.matmul(L,V)-tf.matmul(H,U)+1e-9
		S = tf.divide(R,Z)
		self.R_out = X*tf.matmul(S,tf.transpose(W))-tf.multiply(L,tf.matmul(S,tf.transpose(V))) - tf.multiply(H, tf.matmul(S,tf.transpose(U)))
		return self.R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

class NextLinear(Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lrp_module = modules.NextLinear

	def deep_taylor(self, R):
		self.R_in = R
		V = tf.nn.relu(self.weights)
		Z = tf.matmul(self.input_tensor, V) + 1e-9
		S = tf.divide(R, Z)
		C = tf.matmul(S, tf.transpose(V))
		R_out = tf.multiply(self.input_tensor, C)
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

class FirstLinear(Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lrp_module = modules.FirstLinear

	def deep_taylor(self, R):
		return self.zB_lrp(R)

# -------------------------
# Sum-pooling layer
# -------------------------
class Pooling(Layer):
	def forward(self, input_tensor):
		input_shape = input_tensor.get_shape().as_list()[1:3]
		if input_shape[0] % 2 + input_shape[1] % 2 > 0:
			raise Exception("Input for pooling layer must have even spatial dims, but has "+str(input_shape))
		self.input_tensor = input_tensor
		# alternative implementation for:
		output_tensor = tf.nn.pool(input_tensor, [2, 2], "AVG", "VALID", strides=[2,2])*2
		self.output_tensor = output_tensor
		return output_tensor

	def _simple_lrp(self, R, input_tensor):
		input_tensor += 1e-9
		output_tensor = tf.nn.pool(input_tensor, [2, 2], "AVG", "VALID", strides=[2,2])*2
		R = tf.reshape(R, shape(output_tensor))
		R_per_act = tf.divide(R, output_tensor)
		R_per_in_act = tf.gradients(output_tensor, input_tensor, grad_ys=R_per_act)[0]
		R_out = tf.multiply(R_per_in_act, input_tensor)
		return R_out

	def simple_lrp(self, R):
		R_out = self._simple_lrp(R, self.input_tensor)
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def alphabeta_lrp(self, R, alpha=1.):
		R_out = self._simple_lrp(R, self.input_tensor)
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def alphabeta_lrp_(self, R, alpha=1.):
		input_plus = tf.nn.relu(self.input_tensor)
		input_minus = tf.nn.relu(-self.input_tensor)

		R_plus_out = self._simple_lrp(R, input_plus)
		R_minus_out = self._simple_lrp(R, input_minus)

		R_out = alpha*R_plus_out + (1.-alpha)*R_minus_out
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def deep_taylor(self, R): return self.alphabeta_lrp(R, 1.)

	def to_numpy(self):
		return modules.Pooling()

# -------------------------
# Convolution layer
# -------------------------
class Convolution(AbstractLayerWithWeights):
	"""
	Inherits _simple_lrp based methods, which use self.linear_operation
	"""
	def __init__(self, w_shape):
		self.w_shape = w_shape
		self.linear_operation = lambda input_tensor, weights: tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding="VALID")
		def input_reshape(input_tensor):
			input_shape = input_tensor.get_shape().as_list()
			if len(input_shape) < 4:
				h = np.prod(input_shape[1])/28
				print("Convolutional layer: reshape to height:", h)
				input_tensor = tf.reshape(input_tensor, [-1, int(h), 28, 1])
			return input_tensor
		self.input_reshape = input_reshape
		self.bias_shape = [1, 1, 1, self.w_shape[-1]]
		self.get_w_shape = lambda input_tensor: self.w_shape

		if len(w_shape) != 4:
			raise Exception("Convolutional Layer: w has to be of rank 4, but is {}. Maybe add None as batch_size dim".format(len(w_shape)))

	def gradprop(self, DY, W):
		Y = self.linear_operation(self.input_tensor, W)
		return tf.gradients(Y, self.input_tensor, DY)[0]

	def zB_lrp(self, R):
		W_plus = tf.nn.relu(self.weights)
		W_minus = -tf.nn.relu(-self.weights)
		X,L,H = self.input_tensor,self.input_tensor*0+utils.lowest,self.input_tensor*0+utils.highest
		Z = self.linear_operation(X, self.weights)-self.linear_operation(L, W_plus)-self.linear_operation(H, W_minus)+1e-09
		S = tf.divide(R, Z)

		R_out = X*self.gradprop(S, self.weights)-L*self.gradprop(S, W_plus)-H*self.gradprop(S, W_minus)
		return R_out, tf.reduce_sum(R_out) / tf.reduce_sum(R)

	def ww_lrp(self, R):
		raise NotImplementedError()

class NextConvolution(Convolution):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lrp_module = modules.NextConvolution

	def deep_taylor(self, R):
		return self.alphabeta_lrp(R, alpha=1.)
	
class FirstConvolution(Convolution):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lrp_module = modules.FirstConvolution

	def deep_taylor(self, R):
		return self.zB_lrp(R)

	