from ..lrp.train import *

cnn_1 = [Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
		NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
		NextLinear(1024), ReLU(),
		NextLinear(10)]

cnn_2 = [Format(), FirstConvolution([5, 5, 1, 10]), ReLU(), Pooling(),
		NextConvolution([5, 5, 10, 25]), ReLU(), Pooling(),
		NextConvolution([4, 4, 25, 100]), ReLU(),
		NextConvolution([1, 1, 100, 10]), Flatten()]
		
cnn_3 = [Format(), FirstConvolution([5, 5, 1, 10]), ReLU(), Pooling(),
		NextConvolution([5, 5, 10, 25]), ReLU(), Pooling(),
		NextConvolution([4, 4, 25, 100]), ReLU(),
		NextLinear(10)]
		#NextConvolution([1, 1, 100, 10]), Flatten()]