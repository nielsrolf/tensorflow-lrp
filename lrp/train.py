from . import modules
from . import utils
import numpy as np
import tensorflow as tf
import uuid
from utils import mkdir

import pdb

# -------------------------
# Stopping Criterion
# -------------------------
def gl(alpha, minimum=3000):
    # returns a stopping criterion: train_step, val_accs -> boolean
    def criterion(i, val_accs):
        if len(val_accs) <= 2: return True
        if i < minimum: return True
        opt = np.max(val_accs)
        best_index = val_accs.index(opt)
        if len(val_accs) - best_index >= 10 and len(val_accs) >= 1.5 * best_index: return False
        return (1 - val_accs[-1]) / (1 - opt) <= 1. + alpha / 100

    return criterion


# -------------------------
# Vars and tf helpers
# -------------------------
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


def normalize_batch(R):
    rank = len(shape(R))
    sum_axes = list(range(1, rank))
    R_sum = tf.reduce_sum(R, axis=sum_axes) + 1e-9
    R_norm = tf.divide(R, expand_dims(R_sum, axes=sum_axes))
    return R_norm


def flatten_batch(B):
    return tf.reshape(B, [-1, int(np.prod(B.shape[1:]))])


class namescope():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tf_namescope = tf.name_scope(self.name)
        self.tf_namescope.__enter__()  # if FLAGS.logging else None
        return self.tf_namescope

    def __exit__(self, *args):
        # if FLAGS.logging: self.tf_namescope.__exit__(*args)
        self.tf_namescope.__exit__(*args)


# -------------------------
# Network
# -------------------------
class Network():
    def __init__(self, layers, input_tensor, y_, max_num_instances=10000000, logdir=".temp", loss=None):
        self.format_layer = layers[0]
        self.layers = layers[1:]
        self.input_tensor = input_tensor
        self.X = input_tensor
        self.y_ = y_
        self.session = None
        self.real_layers = []
        self.logdir = logdir
        tf.gfile.MakeDirs(logdir)

        # forward through every layer
        with namescope("format") as scope:
            activation = self.format_layer.forward(input_tensor)

        print("Network layer mappings:", activation.shape)
        for layer_id, layer in enumerate(self.layers):
            with namescope(str(layer_id) + "-" + layer.__class__.__name__):
                activation = layer.forward(activation)
            print("->", activation.shape)
            if not isinstance(layer, (Activation, Format, NoFormat, Flatten)):
                self.real_layers.append(layer_id)

        self.y = activation
        self.sigmoid_y = tf.sigmoid(activation)
        self.normalized_sigmoid = normalize_batch(self.sigmoid_y)

        if loss is None:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        else:
            self.loss = tf.reduce_mean(loss(self.y_, self.y))

        self.correct_classified = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        if shape(self.y)[-1] == 1 and len(shape(self.y)) == 2:
            self.accuracy = 1 - self.loss
        else:
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_classified, tf.float32))

        self.train = tf.train.AdamOptimizer().minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=max_num_instances)

    def maximize_activation(self, layer_id, pattern, step_size=0.01):
        # something like deepdream
        # define objective
        target = flatten_batch(self.layers[layer_id].output_tensor)
        loss = -tf.reduce_sum(tf.multiply(target, pattern))
        # get gradient
        gradient = tf.gradients(loss, self.input_tensor)[0]
        return self.input_tensor - step_size * gradient

    def sensivity(self, class_filter):
        # returns the gradient of class_filter.dot(readout)  wrt the pixel activation
        target = tf.reduce_sum(tf.multiply(self.y, class_filter))
        return tf.gradients(target, self.format_layer.output_tensor)[0]

    def lrp(self, class_filter, methods="simple", explain_layer_id=-1, substract_mean=None, reference=None):
        """
        Gets:
            class_filter: relevance for which class? - as one hot vector; for correct class use ground truth placeholder
            Methods: which method to use? Same meaning as in layerwise_lrp
        Returns:
            R: input layers relevance tensor
        """
        Rs, _ = self.layerwise_lrp(class_filter, methods, explain_layer_id=explain_layer_id,
                                   substract_mean=substract_mean, reference=reference)
        return Rs[0]

    def layerwise_lrp(self, class_filter, methods="simple", method_id=None, consistent=False, explain_layer_id=-1,
                      substract_mean=None, reference=None):  # class_filter: tf.constant one hot-vector
        """
        Gets:
            class_filter: relevance for which class? - as one hot vector; for correct class use ground truth placeholder
            Methods: which method to use?
                If the same method should be used for every layer, then the string can be passed: ("simple" / "ab")
                If one of the following standard method combinations shall be used, also pass the string code:
                    "zbab" -> zb - rule for the first layer, after that ab-rule with alpha=2.
                    "wwab" -> ww - rule for the first layer, after that ab-rule with alpha=2.
                If this is the case, but the methods needs an additional numeric parameter, then it can be passed like
                    ["methodstr", <param>]
                If the methods shall be specified for each layer, then a list has to be passed, where each element is a
                    list like ["methodstr"(, <param>)]
            method_id: an identifier for this stack
            consistent: if True, negative relevance will be filtered out in readout layer
            explain_layer_id: Where to apply class filter. This can be used to explain the activation of a certain
                              vector in any layer
            subtract_mean: for pca explanations: explain (Activation-Mean).dot(class_filter)
            reference: choose reference=None for regular lrp, reference="batch" to use activations-mean_in_
                       current_batch(activations) for lrp calculations and reference=X to subtract avg activations for
                       X for lrp calculations
        Returns:
            R_layerwise: list of relevance tensors, one for each layer, so that R[0] is in input-space and R[-1] is in
                        readout-layer space
        Creates a self.explained_f, use this only with one lrp stack per Network. This can be used to validate the
            conservativeness
            For PC Activation Explanations, the mean of this should be zero except for deeptaylor
        """
        method_id = method_id if not method_id is None else str(uuid.uuid4())[:3]

        a = self.layers[explain_layer_id].output_tensor
        if substract_mean is not None:
            a -= substract_mean
        self.test = tf.reduce_sum(a, axis=0)

        R = tf.multiply(a, class_filter / tf.reduce_sum(tf.nn.relu(class_filter)))
        self.explained_f = tf.reduce_sum(R, axis=list(range(1, len(R.shape))))

        if methods == "deeptaylor" or consistent:
            R = tf.nn.relu(R)

        if reference is not None:
            if isinstance(reference, np.ndarray):
                ref = tf.constant(reference, dtype=tf.float32)
                ref = self.format_layer.forward_silent(ref)
            elif reference == "batch":
                ref = self.format_layer.output_tensor
            refs = []
            for layer in self.layers:
                refs.append(tf.reduce_mean(ref, axis=0, keep_dims=True))
                ref = layer.forward_silent(ref)
        else:
            refs = [0]*len(self.layers)

        if methods == "zbab":
            methods = [["zb"]] + [["ab", 2.]] * (len(self.layers) - 1)
        elif methods == "wwab":
            methods = [["ww"]] + [["ab", 2.]] * (len(self.layers) - 1)
        elif methods == "zb-gdt": # generalized deep taylor
            methods = [["zb"]] + [["gdt", 0.5]] * (len(self.layers) - 1)
        elif isinstance(methods, str):
            methods = [[methods]] * len(self.layers)
        elif isinstance(methods[1], (float, int)):
            methods = [methods] * len(self.layers)
        R_layerwise = [R]
        Conservation_layerwise = [tf.constant(1., dtype=tf.float32)]
        last_layer = explain_layer_id % len(self.layers)
        for l, m, layer_id, ref in zip(self.layers[::-1], methods[::-1], list(range(len(self.layers)))[::-1], refs[::-1]):
            if layer_id > last_layer: continue  # skip the layers that come after the to be explained stuff
            with namescope(str(layer_id) + "-" + l.__class__.__name__ + "-" + m[0] + method_id):
                backward_mapping = str(shape(R)) + " -> "
                if m[0] == "deep_taylor" or m[0] == "deeptaylor":
                    if layer_id > 0:
                        R, C = l.deep_taylor(R)
                    else:
                        R, C = self.format_layer.choose_deeptaylor(l, R)
                elif m[0] == "gdt":
                    R, C = l.generalized_deeptaylor(R, m[1])
                elif m[0] == "simple":
                    R, C = l.simple_lrp(R, ref=ref)
                elif m[0] == "ab" or m[0] == "alphabeta":
                    if len(m) > 1:
                        if isinstance(l, AbstractLayerWithWeights):
                            R, C = l.alphabeta_lrp(R, m[1], ref=ref)
                        else:
                            R, C = l.alphabeta_lrp(R, m[1])
                    else:
                        if isinstance(l, AbstractLayerWithWeights):
                            R, C = l.alphabeta_lrp(R, ref=ref)
                        else:
                            R, C = l.alphabeta_lrp(R)

                elif m[0] == "simpleb":
                    R, C = l.simple_lrp(R)
                elif m[0] == "abb" or m[0] == "alphabeta":
                    R, C = l.alphabeta_lrp(R, m[1])
                elif m[0] == "zb":
                    R, C = l.zB_lrp(R)
                elif m[0] == "ww":
                    R, C = l.ww_lrp(R)
                else:
                    raise Exception("Unknown LRP method: {}".format(m[0]))
                backward_mapping += str(shape(R))
                # print(type(l), ": ", m, ":", backward_mapping)
                R_layerwise = [R] + R_layerwise
                Conservation_layerwise = [C] + Conservation_layerwise
        return R_layerwise, Conservation_layerwise

    def layerwise_conservation_test(self, R_layerwise, Conservation_layerwise, feed_dict):
        if self.sess is None: raise Exception(
            "Network.layerwise_conservation_test must be called while Network.session is set")
        # print("\n")
        r_in = R_layerwise[-1]
        for i_, (l, C, R) in enumerate(
                zip(self.layers[::-1], Conservation_layerwise[:-1][::-1], R_layerwise[:-1][::-1])):
            i = len(self.layers) - 1 - i_
            c, input_array, output_array, r = self.sess.run([C, l.input_tensor, l.output_tensor, R],
                                                            feed_dict=feed_dict)

    # print("Layer {}: {}: Conservation: {}".format(i, type(l), c))
    # print("		Forward:", shape(input_array), " -> ", shape(output_array))
    # print("		Backward", shape(r), " <- ", shape(r_in))
    # print("R_out = ", np.sum(r))

    def layerwise_conservation_test_(self, R_layerwise, Conservation_layerwise, feed_dict):
        if self.sess is None: raise Exception(
            "Network.layerwise_conservation_test must be called while Network.session is set")
        r_in = self.layers[-1]
        conservation_layerwise = self.sess.run(Conservation_layerwise, feed_dict=feed_dict)
        for i, (l, c) in enumerate(zip(self.layers + ["filtered forwarded readout layer = Relevance input layer"],
                                       conservation_layerwise)):
            print("Layer {}: {}: Conservation: {}".format(i, type(l), c))

    def save_params(self, export_dir):
        tf.gfile.MakeDirs(export_dir)
        save_path = self.saver.save(self.sess, "{}/model.ckpt".format(export_dir))

    # print("Saved model to ", save_path)

    def load_params(self, import_dir):
        #self.create_session()
        self.saver.restore(self.sess, "{}/model.ckpt".format(import_dir))

    def reset_params(self):
        self.create_session()

    def close_sess(self):
        try:
            self.sess.close()
        except:
            pass

    def create_session(self):
        self.close_sess()
        summary = tf.summary.merge_all()
        self.set_session(tf.Session())
        # summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
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

    def get_numpy_deeptaylor(self, X, dim):  # dim: one hot vector; heatmaps for correct class: dim = y_
        # X: either np.array or tf.Tensor
        numpy_network = self.to_numpy()
        y = numpy_network.forward(X)
        return numpy_network.relprop(y * dim), y * dim

    def conservation_check(self, heatmaps, R):
        # input as numpy.array
        num_samples = heatmaps.shape[0]
        h = np.reshape(heatmaps, [num_samples, np.prod(heatmaps.shape[1:])])
        hR = np.sum(h, axis=1)
        R = np.reshape(R, hR.shape)
        err = np.absolute(hR - R)

    # print("Relevance - Sum(heatmap) = ", err)

    def ab_test(self, feed_dict):
        ab_errors = self.sess.run([l.ab_forward_error for l in self.layers if isinstance(l, Convolution)],
                                  feed_dict=feed_dict)
        for error in ab_errors:
            print("AB forward error: ", np.mean(np.absolute(error)));
            input()

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

    def fit(self, data, stopping_criterion=gl(300, 5000), perform_action=None, action_rythm=50, val_dict=None,
            validation_rythm=50, train_batch_size=50):
        """
        Fit a dataset using early stopping, with the possibility to inject any action and a rythm into the train process,
         i.e. tracking stuff
        Gets:
            data: a data object that implements next_batch() and validation_batch, expected to return a feed_dict or a
                  (X, y_)-tuple
            perform_action(train_step)
        """
        logdir = ".temp/" + str(uuid.uuid4())[:5]
        mkdir(logdir)
        if val_dict is None:
            batch = data.validation_batch()
            val_dict = batch if isinstance(batch, dict) else {self.input_tensor: batch[0], self.y_: batch[1]}
        val_accs = []
        i = 0
        best_acc = 0
        while stopping_criterion(i, val_accs):
            batch = data.next_batch(train_batch_size)
            feed_dict = batch if isinstance(batch, dict) else {self.input_tensor: batch[0], self.y_: batch[1]}
            self.sess.run(self.train, feed_dict=feed_dict)
            if i % validation_rythm == 0:
                val_accs += [self.sess.run(self.accuracy, feed_dict=val_dict)]
                print("Train step ", i, ": ", val_accs[-1])
                if val_accs[-1] >= best_acc:
                    best_acc = val_accs[-1]
                    self.save_params(logdir)
            if i % action_rythm == 0 and perform_action is not None: perform_action(i)
            i += 1
        self.load_params(logdir)
        acc = self.sess.run(self.accuracy, feed_dict=val_dict)
        print("Training finished with validation accuracy:", acc)
        return val_accs

    def __call__(self, X):
        a = self.format(X)
        for layer in self.layers:
            a = layer(a)
        return a

    def get_mean_var(self, input_mean, input_var):
        """
        Estimate relevance by inspecting the influence of variance in features (unknown features) on the variance of
        predictions
        :param input_mean: tf.placeholder of shape self.input_tensor, but for a single sample
        :param input_var: tf.placeholder of shape self.input_tensor, but for a single sample
        :return: mean and variance of shape self.y
        """
        mean, var = input_mean, input_var
        for layer in [self.format_layer] + self.layers:
            print("mean", mean.shape, "var", var.shape)
            mean, var = layer.forward_mean_var(mean, var)
        return mean, var


# -------------------------
# Abstract Layer
# -------------------------
class Layer():
    def __init__(self):
        self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def to_numpy(self):  # For layers with weights and biases (Linear, Convolution); Pooling and ReLU override this
        if self.sess is None:
            raise Exception("Trying to extract variables without active session")
        else:
            W, B = self.sess.run([self.weights, self.biases], feed_dict={})
            return self.lrp_module(W, B)

    def set_reference(self, reference):
        self.reference = reference

    def forward_mean_var(self, mean, var):
        x_shape = [10000]+[int(d) for d in self.input_tensor.shape[1:] ]
        s_shape = [1] + x_shape[1:]
        mean = tf.reshape(mean, s_shape)
        var = tf.reshape(var, s_shape)
        X = tf.random_normal(x_shape, mean, var)
        y = self.forward_silent(X)
        mean_out, var_out = tf.nn.moments(y, axes=0)
        return mean_out, var_out


# -------------------------
# Format Layer
# -------------------------
class Format(Layer):
    def __init__(self, highest=1., lowest=-1.):
        super().__init__()
        self.highest, self.lowest = highest, lowest

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.add(input_tensor * (self.highest - self.lowest), self.lowest)
        return self.output_tensor

    def __call__(self, a):
        return self.lowest + a * (self.highest - self.lowest)

    def forward_silent(self, input_tensor):
        return tf.add(input_tensor * (self.highest - self.lowest), self.lowest)

    def choose_deeptaylor(self, layer, R):
        return layer.zB_lrp(R)

    def to_numpy(self):
        return modules.Format()

    def lrp(self, R): return R, tf.constant(1.)

    def deep_taylor(self, R): return self.lrp(R)

    def simple_lrp(self, R, ref=0.): return self.lrp(R)

    def alphabeta_lrp(self, R, alpha=1.): return self.lrp(R)


class NoFormat(Format):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = input_tensor
        return self.output_tensor

    def __call__(self, a):
        return a

    def forward_silent(self, input_tensor):
        return input_tensor

    def choose_deeptaylor(self, layer, R):
        return layer.ww_lrp(R)


class Flatten(Format):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.reshape(input_tensor, [-1, np.prod(shape(input_tensor)[1:])])
        return self.output_tensor

    def __call__(self, a):
        return np.reshape(a, [-1, np.prod(shape(a)[1:])])

    def forward_silent(self, input_tensor):
        return tf.reshape(input_tensor, [-1, np.prod(shape(input_tensor)[1:])])

"""
class Whiten(Format):
    def __init__(self, X):
        super().__init__()
        X = np.linalg.svd(X, )
"""


class MNISTShape(Format):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])
        return self.output_tensor

    def __call__(self, a):
        return np.reshape(a, [-1, 28, 28, 1])

    def forward_silent(self, input_tensor):
        return tf.reshape(input_tensor, [-1, 28, 28, 1])


class IMGReshape(Format):
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        preformat = super().forward(input_tensor)
        img_shape = self.img_shape
        channels = img_shape[2] if len(img_shape) == 3 else 1
        self.output_tensor = tf.reshape(preformat, [-1, img_shape[0], img_shape[1], channels])
        return self.output_tensor

    def __call__(self, a):
        a = super().__call__(a)
        img_shape = self.img_shape
        channels = img_shape[2] if len(img_shape) == 3 else 1
        return np.reshape(a, [-1, img_shape[0], img_shape[1], channels])


    def forward_silent(self, input_tensor):
        preformat = super().forward(input_tensor)
        img_shape = self.img_shape
        channels = img_shape[2] if len(img_shape) == 3 else 1
        return tf.reshape(preformat, [-1, img_shape[0], img_shape[1], channels])


# -------------------------
# Activation Layers
# -------------------------
class Activation(Layer):
    def deep_taylor(self, R): return R, tf.constant(1.)

    def simple_lrp(self, R, ref=0.): return R, tf.constant(1.)

    def alphabeta_lrp(self, R, alpha=1.): return R, tf.constant(1.)

    def generalized_deeptaylor(self, R, gamma=0.5): return R, tf.constant(1.)


class ReLU(Activation):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.nn.relu(input_tensor)
        return self.output_tensor

    def __call__(self, a):
        return np.max(0, a)

    def forward_silent(self, input_tensor):
        return tf.nn.relu(input_tensor)

    def to_numpy(self):
        return modules.ReLU()


class Tanh(Activation):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.nn.tanh(input_tensor)
        return self.output_tensor

    def __call__(self, a):
        return np.tanh(a)

    def forward_silent(self, input_tensor):
        return tf.nn.tanh(input_tensor)


class Abs(Activation):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = tf.abs(input_tensor)
        return self.output_tensor

    def __call__(self, a):
        return np.abs(a)

    def forward_silent(self, input_tensor):
        return  tf.abs(input_tensor)

    def to_numpy(self):
        return modules.Abs()


# -------------------------
# Connected Layers
# -------------------------
class AbstractLayerWithWeights(Layer):
    """
    Handles forwarding and _simple_lrp based stuff. Child classes must set the following in __init___
    - input_reshape(input_tensor)
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

    def __call__(self, a):
        w, b = self.sess.run([self.weights, self.biases])
        return self.linear_np(self.input_reshape_np(a), w) + b

    def forward_silent(self, input_tensor):
        activators = self.linear_operation(self.input_reshape(input_tensor), self.weights)
        return activators + self.biases

    def _simple_lrp(self, R, weights, input_tensor):
        activators = self.linear_operation(input_tensor, weights)
        R = tf.reshape(R, tf.shape(activators))
        R_per_act = tf.divide(R, activators + 1e-12)
        R_per_in_act = tf.gradients(activators, input_tensor, grad_ys=R_per_act)[0]
        R_out = tf.multiply(R_per_in_act, input_tensor)
        return R_out  # R_per_in_act#R_out

    def simple_lrp(self, R, ref=0.):
        if ref != 0.:
            input_tensor = self.input_tensor - self.input_reshape(ref)
        else:
            input_tensor = self.input_tensor
        R_out = self._simple_lrp(R, self.weights, input_tensor)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1

    def alphabeta_lrp_(self, R, alpha=1.):
        # WARNING works only if input_tensor is positive
        input_tensor = self.input_tensor + 1e-9
        w_plus = tf.nn.relu(self.weights) + 1e-9
        w_minus = tf.nn.relu(-self.weights) + 1e-9
        R_out_plus = self._simple_lrp(R, w_plus, input_tensor)
        R_out_minus = self._simple_lrp(R, w_minus, input_tensor)
        R_out = alpha * R_out_plus + (1. - alpha) * R_out_minus
        return R_out, tf.reduce_mean(tf.abs(tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1) /
               tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) -1)) + 1

    def alphabeta_lrp(self, R, alpha=1., ref=0.):
        if ref != 0.:
            input_tensor = self.input_tensor - self.input_reshape(ref)
        else:
            input_tensor = self.input_tensor
        # Split activators into + and -
        input_plus = tf.nn.relu(input_tensor) + 1e-9
        input_minus = tf.nn.relu(-input_tensor) + 1e-9
        w_plus = tf.nn.relu(self.weights) + 1e-9
        w_minus = tf.nn.relu(-self.weights) + 1e-9

        a_plus1 = self.linear_operation(input_plus, w_plus)
        a_plus2 = self.linear_operation(input_minus, w_minus)
        a_minus1 = self.linear_operation(input_plus, w_minus)
        a_minus2 = self.linear_operation(input_minus, w_plus)
        # the following can be used to check if activator decomposition holds:
        # self.ab_forward_error = tf.divide(self.activators - (a_plus1+a_plus2 - (a_minus1+a_minus2)), self.activators)

        R_plus1 = tf.multiply(R, tf.divide(a_plus1, a_plus1 + a_plus2))
        R_plus2 = tf.multiply(R, tf.divide(a_plus2, a_plus1 + a_plus2))
        R_minus1 = tf.multiply(R, tf.divide(a_minus1, a_minus1 + a_minus2))
        R_minus2 = tf.multiply(R, tf.divide(a_minus2, a_minus1 + a_minus2))

        R_out_plus = self._simple_lrp(R_plus1, w_plus, input_plus) + self._simple_lrp(R_plus2, w_minus, input_minus)
        R_out_minus = self._simple_lrp(R_minus1, w_minus, input_plus) + self._simple_lrp(R_minus2, w_plus, input_minus)

        R_out = alpha * R_out_plus + (1. - alpha) * R_out_minus

        return R_out, tf.reduce_mean(tf.abs(tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1) /
               tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) -1)) + 1

    def ww_lrp(self, R):
        ww = tf.square(self.weights) + 1e-9
        in_ones = tf.constant(np.ones([d if d > 0 else 1 for d in shape(self.input_tensor)]), tf.float32)
        t = self.linear_operation(
            in_ones,
            ww)
        R_per_t = tf.divide(R, t)
        R_out = tf.gradients(t, in_ones, grad_ys=R_per_t)[0]
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1

    def generalized_deeptaylor(self, R, gamma=0.5):
        w = tf.maximum(self.weights, 0) + gamma*tf.minimum(self.weights, 0)
        R_out = self._simple_lrp(R, w, self.input_tensor)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1


# -------------------------
# Fully-connected layer
# -------------------------
class Linear(AbstractLayerWithWeights):
    """
    A = input*weights + bias
    """
    def __init__(self, num_neurons):
        super().__init__()
        self.linear_operation = tf.matmul
        self.linear_operation_np = lambda a, w: a.dot(w)
        self.output_dims = num_neurons

        def input_reshape(input_tensor):
            input_shape = input_tensor.get_shape().as_list()
            if len(input_shape) == 4:
                input_shape = [-1, np.prod(input_shape[1:])]
                input_tensor = tf.reshape(input_tensor, input_shape)
            return input_tensor

        def input_reshape_np(a):
            input_shape = a.shape
            if len(input_shape) == 4:
                input_shape = [-1, np.prod(input_shape[1:])]
                a = np.reshape(a, input_shape)
            return input_tensor

        self.input_reshape = input_reshape
        self.input_reshape_np = input_reshape_np
        self.bias_shape = [self.output_dims]

        def get_w_shape(input_tensor):
            return [shape(input_tensor)[-1], num_neurons]

        self.get_w_shape = get_w_shape  # lambda input_tensor : [shape(input_tensor)[-1], num_neurons]

    def zB_lrp(self, R):
        self.R_in = R
        W, V, U = self.weights, tf.maximum(0., self.weights), tf.minimum(0., self.weights)
        X, L, H = self.input_tensor, self.input_tensor * 0 + utils.lowest, self.input_tensor * 0 + utils.highest

        Z = tf.matmul(X, W) - tf.matmul(L, V) - tf.matmul(H, U) + 1e-9
        S = tf.divide(R, Z)
        R_out = X * tf.matmul(S, tf.transpose(W)) - tf.multiply(L, tf.matmul(S, tf.transpose(V))) - \
            tf.multiply(H, tf.matmul(S, tf.transpose(U)))
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1


# -------------------------
# Convolution layer
# -------------------------
class Convolution(AbstractLayerWithWeights):
    """
    Inherits _simple_lrp based methods, which use self.linear_operation
    """

    def __init__(self, w_shape, padding="VALID"):
        self.w_shape = w_shape
        self.linear_operation = lambda input_tensor, weights: tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1],
                                                                           padding=padding)
        self.linear_operation_np = lambda a, w: self.sess.run(tf.nn.conv2d(a, w, [1, 1, 1, 1],
                                                                           padding=padding))
        self.padding = padding

        def input_reshape(input_tensor):
            input_shape = input_tensor.get_shape().as_list()
            if len(input_shape) < 4:
                h = np.prod(input_shape[1]) / 28
                # print("Convolutional layer: reshape to height:", h)
                input_tensor = tf.reshape(input_tensor, [-1, int(h), 28, 1])
            return input_tensor

        self.input_reshape = input_reshape
        self.bias_shape = [1, 1, 1, self.w_shape[-1]]
        self.get_w_shape = lambda input_tensor: self.w_shape

        if len(w_shape) != 4:
            raise Exception(
                "Convolutional Layer: w has to be of rank 4, but is {}. Maybe add None as batch_size dim".format(
                    len(w_shape)))

    def gradprop(self, DY, W):
        Y = self.linear_operation(self.input_tensor, W)
        return tf.gradients(Y, self.input_tensor, DY)[0]

    def zB_lrp(self, R):
        W_plus = tf.nn.relu(self.weights)
        W_minus = -tf.nn.relu(-self.weights)
        X, L, H = self.input_tensor, self.input_tensor * 0 + utils.lowest, self.input_tensor * 0 + utils.highest
        Z = self.linear_operation(X, self.weights) - self.linear_operation(L, W_plus) - \
            self.linear_operation(H, W_minus) + 1e-09
        S = tf.divide(R, Z)

        R_out = X * self.gradprop(S, self.weights) - L * self.gradprop(S, W_plus) - H * self.gradprop(S, W_minus)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1

    def ww_lrp(self, R):
        raise NotImplementedError()


class NextConvolution(Convolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lrp_module = modules.NextConvolution

    def deep_taylor(self, R):
        return self.alphabeta_lrp(tf.nn.relu(R), alpha=1.)


class FirstConvolution(Convolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lrp_module = modules.FirstConvolution

    def deep_taylor(self, R):
        return self.zB_lrp(tf.nn.relu(R))


class NextLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lrp_module = modules.NextLinear

    def deep_taylor(self, R):
        self.R_in = tf.nn.relu(R)
        V = tf.nn.relu(self.weights)
        Z = tf.matmul(self.input_tensor, V) + 1e-9
        S = tf.divide(R, Z)
        C = tf.matmul(S, tf.transpose(V))
        R_out = tf.multiply(self.input_tensor, C)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1


class FirstLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lrp_module = modules.FirstLinear

    def deep_taylor(self, R):
        raise Warning(
            "FirstLinear - deeptaylor: not sure which method to use, use zB-rule. If ww-rule is needed, please explicitly specify it")
        return self.zB_lrp(tf.nn.relu(R))


class LinearWithMoreBias(NextLinear):
    def __init__(self, *args, **kwargs):
        self.bias_multiplicator = kwargs.pop("bias_multiplicator", 1)
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor):
        self.input_tensor = self.input_reshape(input_tensor)
        self.weights = weight_variable(self.get_w_shape(self.input_tensor))
        self.biases = bias_variable(self.bias_shape)
        self.activators = self.linear_operation(self.input_tensor, self.weights)
        self.output_tensor = self.activators + self.biases * self.bias_multiplicator
        return self.output_tensor

    def forward_silent(self, input_tensor):
        input_tensor = self.input_reshape(input_tensor)
        activators =  self.linear_operation(input_tensor, self.weights)
        return activators + self.biases * self.bias_multiplicator



# -------------------------
# Abstract Layer without weights and biases, for Pooling, Padding
# -------------------------
class AbstractLayerWithoutWB(Layer):
    """
    Child class has to implement:
    - self.linear_operation(input_tensor)
    """

    def forward(self, input_tensor):
        self.input_tensor = self.input_reshape(input_tensor)
        # here was a big mistake in earlier versions, this would always implement a certain pooling layer
        self.output_tensor = self.linear_operation(input_tensor)
        return self.output_tensor

    def forward_silent(self, input_tensor):
        input_tensor = self.input_reshape(input_tensor)
        return self.linear_operation(input_tensor)

    def _simple_lrp(self, R, input_tensor):
        input_tensor += 1e-9
        output_tensor = self.linear_operation(input_tensor)
        R = tf.reshape(R, shape(output_tensor))
        R_per_act = tf.divide(R, output_tensor)
        R_per_in_act = tf.gradients(output_tensor, input_tensor, grad_ys=R_per_act)[0]
        R_out = tf.multiply(R_per_in_act, input_tensor)
        return R_out

    def simple_lrp(self, R, ref=0.):
        R_out = self._simple_lrp(R, self.input_tensor)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1

    def alphabeta_lrp(self, R, alpha=1.):
        R_out = self._simple_lrp(R, self.input_tensor)
        return R_out, tf.reduce_mean(tf.abs(
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R_out)), axis=1)+ 1e-9) / 
                                    (tf.reduce_sum(flatten_batch(tf.reduce_sum(R)), axis=1) + 1e-9) -1)) + 1

    def deep_taylor(self, R): return self.alphabeta_lrp(tf.nn.relu(R), 1.)


# -------------------------
# Pooling layers
# -------------------------

class Pooling(AbstractLayerWithoutWB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_operation = lambda input_tensor: tf.nn.pool(input_tensor, [2, 2], "AVG", "SAME", strides=[2, 2]) * 2

        def input_reshape(input_tensor):
            input_shape = input_tensor.get_shape().as_list()[1:3]
            # if input_shape[0] % 2 + input_shape[1] % 2 > 0:
            #	raise Exception("Input for pooling layer must have even spatial dims, but has "+str(input_shape))
            return input_tensor

        self.input_reshape = input_reshape

    def __call__(self, X):
        return 0.5 * (X[:, ::2, ::2, :] + X[:, ::2, 1::2, :] + X[:, 1::2, ::2, :] + X[:, 1::2, 1::2, :])


    def to_numpy(self):
        return modules.Pooling()


class MaxPooling(AbstractLayerWithoutWB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_operation = lambda input_tensor: tf.nn.pool(input_tensor, [2, 2], "MAX", "SAME", strides=[2, 2])
        self.input_reshape = lambda input_tensor: input_tensor

    def __call__(self, a):
        raise NotImplementedError



