import numpy,PIL,PIL.Image

lowest = -1.0
highest = 1.0

# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap_1(x):
	x = x[...,numpy.newaxis]
	plus = numpy.clip(x, 0, 1)
	minus = numpy.clip(x, -1, 0)*(-1)
	neutral = 1 - plus - minus
	print("Minus elem [{}, {}]".format(numpy.amin(minus), numpy.amax(minus)))
	
	s = 0.5

	r = plus
	y = neutral*s
	w = neutral*(1-s)
	b = minus

	r = 1-minus #numpy.clip(r+y + w, 0, 1) # r+y+w = plus + neutral = plus + 1 - plus - minus ) 1 - minus
	g = 1-plus-minus#y + w 
	b = 1-s + s*minus -(1-s)*plus #minus+(1-s)*(1-plus-minus) #b + w
	for f in [r,g,b]:
		print("color range: ", numpy.amin(f), numpy.amax(f))

	return numpy.concatenate([r,g,b],axis=-1)

def heatmap(x):

	x = x[...,numpy.newaxis]
	
	r, g, b = 1., 1., 1.
	max_b = 0.3
	max_r = 0.7
	# positive relevance
	plus = numpy.clip(x, 0., max_r)/max_r
	minus = numpy.clip(-x, 0., max_b)/max_b
	r -= minus 
	g = g - minus - plus
	b -= plus

	return numpy.concatenate([r,g,b],axis=-1)

def heatmap_(x):

	x = x[...,numpy.newaxis]

	r = 0.9 - numpy.clip(x-0.3,0,0.7)/0.7*0.5
	g = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4
	b = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4

	return numpy.concatenate([r,g,b],axis=-1)

def graymap(x):

	x = x[...,numpy.newaxis]
	return numpy.concatenate([x,x,x],axis=-1)*0.5+0.5

# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x,colormap,name):
	print("Visualize", name)
	N = len(x); assert(N<=16)

	x = colormap(x/(numpy.abs(x).max()+1e-09))

	# Create a mosaic and upsample
	x = x.reshape([1,N,28,28,3])
	x = numpy.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
	x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
	x = numpy.kron(x,numpy.ones([2,2,1]))

	PIL.Image.fromarray((x*255).astype('byte'),'RGB').save(name)
