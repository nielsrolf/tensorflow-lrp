import numpy
from . import utils
import copy

# -------------------------
# Feed-forward network
# -------------------------
class Network:

	def __init__(self,layers):
		self.layers = layers

	def forward(self,Z):
		for l in self.layers:
			print("\n", type(l), ":", Z.shape)
			Z = l.forward(Z)
			print("						-> ", Z.shape)
		return Z

	def gradprop(self,DZ):
		for l in self.layers[::-1]: DZ = l.gradprop(DZ)
		return DZ

	def relprop(self,R):
		print("---------------------------\nRelevance Layerwise")
		for l in self.layers[::-1]:
			print("Current layer relavance shape: ", R.shape)
			print("Activation shape of target layer: ", l.A.shape)
			print("Raw R", R.shape)
			R = l.relprop(R)
			#R.reshape(l.A.shape)
			#print("Reshaped R", R.shape)
		print(R.shape)
		return R

# -------------------------
# ReLU activation layer
# -------------------------
class ReLU:

	def forward(self,X):
		self.Z = X>0
		self.A = X*self.Z
		return self.A

	def gradprop(self,DY):
		return DY*self.Z

	def relprop(self,R): return R

# -------------------------
# Fully-connected layer
# -------------------------
class Linear:

	def __init__(self, W, B):
		self.W = W
		self.B = B

	def forward(self,X):
		if len(X.shape) > 2:
			X = numpy.reshape(X, [X.shape[0], numpy.prod(X.shape[1:])])
		self.X = X
		self.A = numpy.dot(self.X,self.W)+self.B
		return self.A

	def gradprop(self,DY):
		self.DY = DY
		return numpy.dot(self.DY,self.W.T)

class NextLinear(Linear):
	def relprop(self,R):
		V = numpy.maximum(0,self.W)
		Z = numpy.dot(self.X,V)+1e-9; S = R/Z
		C = numpy.dot(S,V.T)
		R = self.X*C
		return R

class FirstLinear(Linear):
	def relprop(self,R):
		W,V,U = self.W,numpy.maximum(0,self.W),numpy.minimum(0,self.W)
		X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest
		Z = numpy.dot(X,W)-numpy.dot(L,V)-numpy.dot(H,U)+1e-9
		S = R/Z
		R = X*numpy.dot(S,W.T)-L*numpy.dot(S,V.T)-H*numpy.dot(S,U.T)
		return R

# -------------------------
# Sum-pooling layer
# -------------------------
class Pooling:

	def forward(self,X):
		self.X = X
		self.Y = 0.5*(X[:,::2,::2,:]+X[:,::2,1::2,:]+X[:,1::2,::2,:]+X[:,1::2,1::2,:])
		self.A = self.Y
		return self.Y

	def gradprop(self,DY):
		self.DY = DY
		DX = self.X*0
		for i,j in [(0,0),(0,1),(1,0),(1,1)]: DX[:,i::2,j::2,:] += DY*0.5
		return DX

	def relprop(self,R):
		R = numpy.reshape(R, self.A.shape)
		Z = (self.forward(self.X)+1e-9); S = R / Z
		C = self.gradprop(S);			R = self.X*C
		return R

# -------------------------
# Convolution layer
# -------------------------
class Convolution:

	def __init__(self,W, B):
		self.W = W
		self.B = B

	def forward(self,X):

		self.X = X
		mb,wx,hx,nx = X.shape
		ww,hw,nx,ny = self.W.shape
		wy,hy	   = wx-ww+1,hx-hw+1

		Y = numpy.zeros([mb,wy,hy,ny],dtype='float32')

		for i in range(ww):
			for j in range(hw):
				Y += numpy.dot(X[:,i:i+wy,j:j+hy,:],self.W[i,j,:,:])

		self.A = Y+self.B
		return self.A

	def gradprop(self,DY):

		self.DY = DY
		mb,wy,hy,ny = DY.shape
		ww,hw,nx,ny = self.W.shape

		DX = self.X*0

		for i in range(ww):
			for j in range(hw):
				DX[:,i:i+wy,j:j+hy,:] += numpy.dot(DY,self.W[i,j,:,:].T)

		return DX

class NextConvolution(Convolution):
	def relprop(self,R):
		pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum(0,pself.W)

		Z = pself.forward(self.X)+1e-9; S = R/Z
		C = pself.gradprop(S);		  R = self.X*C
		return R

class FirstConvolution(Convolution):

	def forward(self,X):
		if len(X.shape) < 4:
			h = X.shape[1]/28
			print("FirstConvolution: reshape to height:", h)
			X = numpy.reshape(X, [X.shape[0], int(h), 28, 1])
		return super().forward(X)

	def relprop(self,R):
		iself = copy.deepcopy(self); iself.B *= 0
		nself = copy.deepcopy(self); nself.B *= 0; nself.W = numpy.minimum(0,nself.W)
		pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum(0,pself.W)
		X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest

		Z = iself.forward(X)-pself.forward(L)-nself.forward(H)+1e-9; S = R/Z
		R = X*iself.gradprop(S)-L*pself.gradprop(S)-H*nself.gradprop(S)
		return R