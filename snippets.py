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