With this modul you can define, train, save and load neural network, and use various LRP explaining techniques.
All of this is possible with tensorflow as backenend, and the tensorflow network can be exported to a identical network with numpy as backend. The numpy implementation only has limited features though, namely forwarding and deep taylor LRP.

## Use

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
- Convolutional Layers := FirstConvolutional ( + ReLU + Pooling) (+ Next Convolutional Layers)
- NextConvolutional Layers := NextConvolutional ( + ReLU + Pooling) (+ Next Convolutional Layers)
- Linear Layers := FirstLinear (+ Next Linear Layers)
- Next Linear Layers := NextLinar (+ Next Linear Layers)

The loss function is hardcoded in train.py and per default set to:
loss: cross_entropy(sigmoid(y), y_)
optimizer: Adam

## LRP Calculation:
Network has a member function ´heatmaps(self, X)´

## Hardcoded restrictions, which you are encouraged to edit:
- Loss function and optimizer (Network.__init__())
- If input has to be reshaped for FirstConvolutional, then the spatial dimension of the input is [28*n, 28]. This suits, if you give concatinated mnist images as input

## Coding Conventions
In the tensorflow Network, every layer has an ´input_tensor´ and an ´output_tensor´. The ´input_tensor´ contains the actual input (output of previous layer) and formatting (e.g. reshaping). ´input_tensor.eval()´ should suit as input for numpylayer.forward()
