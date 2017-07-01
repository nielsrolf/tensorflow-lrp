class Pooling(Layer):
	def forward(self, input_tensor):
		input_shape = input_tensor.get_shape().as_list()[1:3]
		if input_shape[0] % 2 + input_shape[1] % 2 > 0:
			raise Exception("Input for pooling layer must have even spatial dims, but has "+str(input_shape))
		self.input_tensor = input_tensor
		# alternative implementation for:
		# self.output_tensor = tf.nn.pool(input_tensor, [2, 2], "AVG", "VALID", strides=[2,2])*2
		# using the form output = tensordor(input, weights)
		_, h, w, c = self.input_tensor.shape
		h, w, c = int(h), int(w), int(c)
		weights = np.zeros([h, w, int(h/2), int(w/2)], dtype=np.float32)
		for i,j in [[0,0], [0,1], [1,0], [1,1]]:
			for h_ in range(int(h/2)):
				for w_ in range(int(w/2)):
					weights[i+2*h_,j+2*w_,h_,w_] = 0.5
		self.weights = tf.constant(weights)
		output_tensor = tf.einsum('ijkl,jkmn->imnl', self.input_tensor, self.weights)
		print(shape(output_tensor))
		self.output_tensor = output_tensor
		return output_tensor

	def simple_lrp(self, R):
		R = tf.reshape(R, shape(self.output_tensor))
		activators_ = tf.einsum('ijkl,jkmn->ijkmnl', self.input_tensor, self.weights)
		R_per_act = tf.divide(R, self.output_tensor+1e-9)  #imnl
		R_out = tf.einsum('ijkmnl,imnl->ijkl', activators_, R_per_act)
		self.R_simple = R_out 
		self.conservation = tf.reduce_sum(R) / tf.reduce_sum(R_out)
		return R_out

	def alphabeta_lrp(self, R, alpha=1.):
		print("R: ", R.shape, " -> ", self.output_tensor.shape); input()
		R = tf.reshape(R, shape(self.output_tensor))
		activators_ = tf.einsum('ijkl,jkmn->ijkmnl', self.input_tensor, self.weights)

		activators_plus_ = tf.nn.relu(activators_)
		output_plus = tf.einsum('ijkmnl->imnl', activators_plus_)
		R_per_act_plus = tf.divide(R, output_plus)  #imnl
		R_plus_out = tf.einsum('ijkmnl,imnl->ijkl', activators_plus_, R_per_act_plus)

		activators_minus_ = tf.nn.relu(-activators_)
		output_minus = tf.einsum('ijkmnl->imnl', activators_minus_)
		R_per_act_minus = tf.divide(R, output_minus)  #imnl
		R_minus_out = tf.einsum('ijkmnl,imnl->ijkl', activators_minus_, R_per_act_minus)

		return alpha*R_plus_out + (1--alpha)*R_minus_out

class Convolution():
	def _simple_lrp(self, R):
		R = tf.reshape(R, shape(self.output_tensor))

		w_w, h_w, _, _ = shape(self.weights); w_w, h_w = int(w_w), int(h_w)
		padding = np.array([[0,0], [w_w-1, w_w-1], [h_w-1, h_w-1], [0,0]])
		R_ = tf.pad(R, padding)

		wT = tf.einsum('ijkl->jilk', self.weights)
		normalizer = tf.einsum('jilk->jik', wT)
		wT_normal = tf.divide(wT, tf.expand_dims(normalizer, axis=2))
		R_per_in_act = self.conv2d(R_, wT_normal)
		R_out = tf.multiply(self.input_tensor, R_per_in_act)
		
		self.R_simple = R_out 
		self.conservation = tf.reduce_sum(R) / tf.reduce_sum(R_out)
		return R_out

	def gradprop(self,DY,W):
		mb,wy,hy,ny = DY.shape
		ww,hw,nx,ny = W.shape
		DX = self.input_tensor*0
		for i in range(ww):
			for j in range(hw):
				W_ = tf.transpose(W[i,j,:,:])
				DX[:,i:i+wy,j:j+hy,:] += tf.tensordot(DY, W_, axes=1)
		return DX

	def _simple_lrp(self, R):
		R = tf.reshape(R, shape(self.output_tensor))
		R_per_act_ = tf.divide(R, self.output_tensor)
		"""
		w_w, h_w, _, _ = shape(self.weights); w_w, h_w = int(w_w), int(h_w)
		padding = np.array([[0,0], [w_w-1, w_w-1], [h_w-1, h_w-1], [0,0]])
		R_per_act = tf.pad(R_per_act_, padding)

		wT = tf.einsum('ijkl->jilk', self.weights)
		R_per_in_act = self.conv2d(R_per_act, wT)
		R_out = tf.multiply(self.input_tensor, R_per_in_act)
		"""
		input_shape = shape(self.input_tensor)
		print("Input shape: ", input_shape, type(input_shape))
		R_per_in_act = tf.nn.conv2d_transpose(R_per_act_, self.weights, input_shape, strides=[1,1,1,1], padding="VALID")
		print("R_per_in_act", R_per_in_act.shape); input()
		R_out = tf.multiply(self.input_tensor, R_per_in_act)
		self.R_simple = R_out 
		self.conservation = tf.reduce_sum(R) / tf.reduce_sum(R_out)
		return R_out