# Todo:
	- implement cn like memory cheap simple/ab rule for linear layers
	- implement w^2 rule
	- add epsilon in correct way to ab rule
	
	- Max Pooling Layer 
	- Padding Layers 
	- Variable Strides for CN layers

# Main features
- Define convolutional neural networks and multilayer perceptrons
- Train, save and load the networks
- Explain their decisions with one of the following LRP methods:
	- Simple
	- Alphabeta
	- Deep Taylor
- Visualize filters of the first convolutional layer 
- Export the network to a (numpy implementation)[http://www.heatmapping.org/tutorial/], which is used as reference for testing

With this modul you can define, train, save and load neural network, and use various LRP explaining techniques.
All of this is possible with tensorflow as backenend, and the network can be exported to an identical network with numpy as backend. The numpy implementation only has limited features though, namely forwarding and deep taylor LRP.


## Use

There are some examples in examples.py, which should be mostly self explaining.

Define some input:
Currently some preformatting layer is hardcoded, which assumes the input is in range [0,1] and shall be scretched to range [utils.lowest, utils.highest].
If you want to change that, you can define an alternative for Format(Layer), which could for example implement the identity.

```
	mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
	
	X = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

Define your network architecture, add a session to that network and make it an actual instance of that architecture by that, train it, save it, load it again to see how it works.

```
	mlp = Network([FirstLinear(300), ReLU(), NextLinear(100), ReLU(), NextLinear(10)], X, y_)

	sess = mlp.create_session()

	for i in range(2000):
		acc, _ = sess.run([mlp.accuracy, mlp.train], feed_dict=mlp.feed_dict(mnist.train.next_batch(50)))
		print(acc)
	mlp.save_params("mlp_parameters") # dir name of your choice, will be created
	mlp.load_params("mlp_parameters")
	acc, _ = sess.run([mlp.accuracy, mlp.train], feed_dict=mlp.feed_dict(mnist.test.next_batch(200)))
	print("Finally: ", acc)
```

Calculate and plot some heatmaps: first, we need to get the samples for that the heatmaps shall be calculated, then we calculated them, then we plot them:

```
	X, T = mnist.train.next_batch(10)

	heatmaps, _ = mlp.get_numpy_deeptaylor(X, T)
	utils.visualize(X, utils.heatmap, "yiha/x.png")
	utils.visualize(heatmaps, utils.heatmap, "yiha/correct_class.png")

	E = np.eye(10)
	for c, e in enumerate(E):
		heatmaps, _ = mlp.get_heatmaps(X, e)
		utils.visualize(heatmaps, utils.heatmap, "yiha/class_{}.png".format(c))
	heatmaps, _ = mlp.get_heatmaps(X, np.zeros(10))
	utils.visualize(heatmaps, utils.heatmap, "yiha/nothing.png".format(c))
```

Finally, close the session:
```
	mlp.close_sess()
```

If you implemented a new type of layer and want to check if your .to_numpy() works, you can use network.test():

```
	mlp.test()
```
It checks during the forwarding, if the activation in the numpy network and the tensorflow network are the same.

If you implemented a new relevance propagation rule, you can check the conservation:

```
	...
	X, T = mnist.train.next_batch(10)
	heatmaps, R = mlp.get_heatmaps(X, T)
	mlp.conservation_check(heatmaps, R) # R=network.y.eval()*T
```
	

## Network Architecture and Training
This module allows to define, train and load neural networks with some restricted architectures:

- Network := Convolutional Layer + Next Linear Layers | Linear Layers
- Convolutional Layers := FirstConvolutional ( + Activation + Pooling) (+ Next Convolutional Layers) | (Convolutional ( + Activation + Pooling))*
- NextConvolutional Layers := NextConvolutional ( + Activation + Pooling) (+ Next Convolutional Layers)
- Linear Layers := FirstLinear (+Activation) (+ Next Linear Layers) | (Linear (+ Activation))*
- Next Linear Layers := NextLinar (+Activation) (+ Next Linear Layers)

You don't need to use First- and Next- Linear/Convolutional Layer, but it sets some default settings: you can then use "deeptaylor" as LRP method, which will be translated to zB-rule in the first layer and alphabeta-rule with alpha=1 in the upper layers. It is also necessary if for testing purpose the tf network shall be exported to the numpy network from the tutorial. For training, forwarding, and layerwise specified LRP it is not necessary.

The loss function is hardcoded in train.py and per default set to:
loss: cross_entropy(sigmoid(y), y_)
optimizer: Adam

## LRP Calculation:
The functions `Network.lrp(self, class_filter, methods = "simple")` and `Network.layerwise_lrp(self, class_filter, methods = "simple")`  can be used to generate heatmaps. They are explained in the source code:

```
	def lrp(self, class_filter, methods = "simple"):
		"""
		Gets:
			class_filter: relevance for which class? - as one hot vector; for correct class use ground truth placeholder
			Methods: which method to use? Same meaning as in layerwise_lrp
		Returns:
			R: input layers relevance tensor
		"""
		...

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

``` 

## Filter Visualization
There is another file `filter_visualizer.py` which has nothing to do with LRP, but can visualize the filters learned of the first convolutional layer. In that file there is are some examples, and there is a linear perceptron implemented in two ways, namely as one-layer FC Nńetwork without nonlinearity and as one Layer CNN with filter-size=input-size, and 10 output channels. The visualization shows that they are indeed the same.

## Hardcoded restrictions, which you are encouraged to edit:
- Loss function and optimizer (Network.__init__())
- If input has to be reshaped for FirstConvolutional, then the spatial dimension of the input is [28*n, 28]. This suits, if you give concatinated mnist images as input

## Further restrictions, which take more effort to edit
- Pooling layer is sum pooling
- Pooling Layer and Convolutional Layer don't add any padding. I think the best way to change that is to implement a padding layer, so that the LRP stuff of those two layers doesn't need to be touched.


## Coding Conventions
In the tensorflow Network, every layer has an ´input_tensor´ and an ´output_tensor´. The ´input_tensor´ contains the input (output of previous layer) after formatting (e.g. reshaping).
