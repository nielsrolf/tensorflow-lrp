import PIL,PIL.Image
import numpy as np

lowest = -1.0
highest = 1.0



def soften(x, hardness=1.):
	# hardness <1: softens, = 1 -> no change, >1: sharpens
	s = x*hardness
	_, DX, DY, _ = x.shape
	print("s ", np.sum(s))
	a = x*((1.-hardness)/8.)
	for dx in range(-1,2):
		for dy in range(-1, 2):
			if dx==dy and dy==0: continue
			print(dx, dy)
			s[:,max(0, dx):DX-max(0, -dx), max(0, dy):DY-max(0, -dy),:] += a[:,max(0, -dx):DX-max(0, dx), max(0, -dy):DY-max(0, dy),:]
			print("added ", np.sum(a))
	print("Soften: c = ", np.sum(s)/np.sum(x))

	print("s ", np.sum(s))
	return s

	"""
	for i in range(6):
		hardness = 0.5+i*0.1
		h = lambda x: heatmap(x, hardness)
		visualize_(x, h, name+"_"+str(hardness))
	"""
	
# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap(x, num_soften=0):
	for i in range(num_soften):
		x = soften(x, 0.8)

	x = x[...,np.newaxis]
	r, g, b = 1., 1., 1.
	max_b = 0.2
	max_r = 0.7
	# positive relevance
	plus = np.clip(x, 0., max_r)/max_r
	minus = np.clip(-x, 0., max_b)/max_b
	r -= minus 
	g = g - minus - plus
	b -= plus

	return np.concatenate([r,g,b],axis=-1)

def heatmap_1(x):
	x = x[...,np.newaxis]
	plus = np.clip(x, 0, 1)
	minus = np.clip(x, -1, 0)*(-1)
	neutral = 1 - plus - minus
	print("Minus elem [{}, {}]".format(np.amin(minus), np.amax(minus)))
	
	s = 0.5

	r = plus
	y = neutral*s
	w = neutral*(1-s)
	b = minus

	r = 1-minus #np.clip(r+y + w, 0, 1) # r+y+w = plus + neutral = plus + 1 - plus - minus ) 1 - minus
	g = 1-plus-minus#y + w 
	b = 1-s + s*minus -(1-s)*plus #minus+(1-s)*(1-plus-minus) #b + w
	for f in [r,g,b]:
		print("color range: ", np.amin(f), np.amax(f))

	return np.concatenate([r,g,b],axis=-1)

def heatmap_original(x):

	x = x[...,np.newaxis]

	r = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
	g = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
	b = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

	return np.concatenate([r,g,b],axis=-1)

def graymap(x):

	x = x[...,np.newaxis]
	return np.concatenate([x,x,x],axis=-1)*0.5+0.5

# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x,colormap,name):
	N = len(x); assert(N<=16)

	x = colormap(x/(np.abs(x).max()+1e-09))

	# Create a mosaic and upsample
	x = x.reshape([1,N,28,28,3])
	x = np.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
	x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
	x = np.kron(x,np.ones([2,2,1]))

	PIL.Image.fromarray((x*255).astype('byte'),'RGB').save(name+".png")
