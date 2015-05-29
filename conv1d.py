import numpy as np
import numba
import theano
from theano import tensor as T
import timeit
from theano.tensor.nnet.conv import conv2d
import theano.sandbox.neighbours as TSN


def convolve1d_2D_numpy(a, b, mode='full'):
  nwords, ndim = a.shape
  filter_width, ndim = b.shape
  b = np.flipud(b)  # flip the kernel
  if mode == 'full':
    pad = np.zeros((filter_width-1, ndim))
    a = np.vstack([pad, a, pad])
    shape = (nwords+filter_width-1, filter_width, ndim)
  elif mode == 'valid':
    shape = (nwords-filter_width+1, filter_width, ndim)

  strides = (a.strides[0],) + a.strides
  view = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

  conv_out = np.einsum('kij,ij->kj', view, b)
  return conv_out


class Convolve1d(theano.Op):
  def __init__(self, mode='full'):
    self.mode = mode

  def make_node(self, image, filt):
    image = theano.tensor.as_tensor_variable(image)
    filt = theano.tensor.as_tensor_variable(filt)
    assert image.ndim == 2
    assert filt.ndim == 2
    # assert image.shape[1] == filt.shape[1]
    return theano.Apply(self, [image, filt], [image.type()])

  def make_thunk(self, node, storage_map, compute_map, no_recycling):
    in1_type = getattr(numba, node.inputs[0].dtype)
    in2_type = getattr(numba, node.inputs[1].dtype)
    out_type = getattr(numba, node.outputs[0].dtype)
    self.numba_fct = numba.jit(out_type[:, :](in1_type[:, :], in2_type[:, :]))(convolve1d_2D_numpy)
    # self.numba_fct = convolve1d_2D_numpy
    return super(Convolve1d, self).make_thunk(
        node, storage_map, compute_map, no_recycling)

  def perform(self, node, inputs, outputs):
    image, filt = inputs
    out = self.numba_fct(image, filt, self.mode)
    outputs[0][0] = out

  def infer_shape(self, node, in_shapes):
    nwords, ndim = in_shapes[0]
    filter_width, ndim = in_shapes[1]
    if self.mode == 'full':
      return [(nwords+filter_width-1, ndim)]
    elif self.mode == 'valid':
      return [(nwords-filter_width+1, ndim)]

  def R_op(self, inputs, eval_points):
    rval = None
    if eval_points[0] is not None:
      rval = self.make_node(eval_points[0], inputs[1]).outputs[0]
    if eval_points[1] is not None:
      if rval is None:
        rval = self.make_node(inputs[0], eval_points[1]).outputs[0]
      else:
        rval += self.make_node(inputs[0], eval_points[1]).outputs[0]
    return [rval]

  def grad(self, inputs, output_grads):
    image, filt = inputs
    [gi] = output_grads

    # Wrong gradient, but produces good results
    gi_reverse = gi[::-1]
    out_image = convolve1d_2D(gi_reverse, filt, mode='valid')[::-1]
    out_filt = convolve1d_2D(gi_reverse, image, mode='valid')[::-1]
    return [out_image, out_filt]


def convolve1d_2D(image, filt, mode='full'):
  return Convolve1d(mode)(image, filt)


def convolve1d_4D(input, W, mode='full'):
    batch_size, nchannels, nwords, ndim = input.shape
    nkernels_out, nkernels_in, filter_width, ndim = W.shape
    # Unroll filter along columns
    W_unrolled = W.dimshuffle(0, 2, 1, 3).flatten(ndim=3)

    # Replicate input filters 'batch_size' times and squash out_filters along column axis.
    # W_tiled = T.tile(W_unrolled, (1, 1, batch_size)).dimshuffle(1, 0, 2).flatten(ndim=2)  # doesn't give a gradient
    W_tiled = T.alloc(W_unrolled, batch_size, W_unrolled.shape[0], W_unrolled.shape[1], W_unrolled.shape[2]).dimshuffle(1, 2, 0, 3).flatten(ndim=3).dimshuffle(1, 0, 2).flatten(ndim=2)

    # Unroll input and pad to fit the output filters.
    input_reshaped = input.dimshuffle(0, 2, 1, 3).flatten(ndim=3).dimshuffle(1,0,2).flatten(ndim=2)
    # input_tiled = T.tile(input_reshaped, (1, nkernels_out))
    input_tiled = T.alloc(input_reshaped, nkernels_out, input_reshaped.shape[0], input_reshaped.shape[1]).dimshuffle(1, 0, 2).flatten(ndim=2)

    conv_res = convolve1d_2D(input_tiled, W_tiled, mode=mode)
    if mode == 'full':
      new_shape = (nwords+filter_width-1, nkernels_out, batch_size, nkernels_in, ndim)
    elif mode == 'valid':
      new_shape = (nwords-filter_width+1, nkernels_out, batch_size, nkernels_in, ndim)

    conv_out = conv_res.reshape(new_shape).dimshuffle(2, 1, 0, 3, 4).sum(axis=3)
    return conv_out

##########################################
### Using einsum for 4d matrices
##########################################

def convolve1d_4D_numpy(a, b, mode='full'):
  nbatches, nkernels_in, nwords, ndim = a.shape
  nkernels_out, _, filter_width, _ = b.shape
  b = b[:,:,::-1,:]  # flip
  if mode == 'full':
    pad = np.zeros((nbatches, nkernels_in, filter_width-1, ndim))
    a = np.concatenate([pad, a, pad], axis=2)
    shape = (nbatches, nkernels_in, nwords+filter_width-1, filter_width, ndim)
  elif mode == 'valid':
    shape = (nbatches, nkernels_in, nwords-filter_width+1, filter_width, ndim)

  strides = a.strides[:2] + (a.strides[2],) + a.strides[2:]
  view = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
  conv_out = np.einsum('kqlij,fqij->kflj', view, b)
  return conv_out


class Convolve1d_4D(theano.Op):
    def __init__(self, mode='full'):
      self.mode = mode

    def make_node(self, image, filt):
        image = theano.tensor.as_tensor_variable(image)
        filt = theano.tensor.as_tensor_variable(filt)
        assert image.ndim == 4
        assert filt.ndim == 4
        return theano.Apply(self, [image, filt], [image.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        in1_type = getattr(numba, node.inputs[0].dtype)
        in2_type = getattr(numba, node.inputs[1].dtype)
        out_type = getattr(numba, node.outputs[0].dtype)
        self.numba_fct = numba.jit(out_type[:,:,:,:](in1_type[:,:,:,:],
                                   in2_type[:,:,:,:]))(convolve1d_4D_numpy)
        # self.numba_fct = convolve1d_4D_numpy
        return super(Convolve1d_4D, self).make_thunk(
            node, storage_map, compute_map, no_recycling)

    def perform(self, node, inputs, outputs):
        image, filt = inputs
        out = self.numba_fct(image, filt, self.mode)
        # out = T.patternbroadcast(out, (False, True, False, False))
        outputs[0][0] = out

    def infer_shape(self, node, in_shapes):
        nbatches, nkernels_in, nwords, ndim = in_shapes[0]
        nkernels_out, nkernels_in, filter_width, ndim = in_shapes[1]
        if self.mode == 'full':
          return [(nbatches, nkernels_out, nwords+filter_width-1, ndim)]
        elif self.mode == 'valid':
          return [(nbatches, nkernels_out, nwords-filter_width+1, ndim)]

    def grad(self, inputs, output_grads):
      image, filt = inputs

      # We need to reverse the gradients along axis=2 (nwords),
      # compute convolution, then reverse again.
      reverse_slicing = [slice(None, None, None)] * filt.ndim
      reverse_slicing[2] = slice(None, None, -1)
      reverse_slicing = tuple(reverse_slicing)

      ## TODO :: make sure the gradient is correct
      [gi] = output_grads
      gi_shuffled = gi.dimshuffle(1, 0, 2, 3)
      filt_sh = filt.dimshuffle(1, 0, 2, 3)
      image_sh = image.dimshuffle(1, 0, 2, 3)
      out_image = convolve1d_4D_einsum(gi[reverse_slicing], filt_sh, mode='valid')[reverse_slicing]
      out_filt = convolve1d_4D_einsum(gi_shuffled[reverse_slicing], image_sh, mode='valid')[reverse_slicing]

      return [out_image, out_filt]



def convolve1d_4D_einsum(image, filt, mode='full'):
  return Convolve1d_4D(mode=mode)(image, filt)

######

def convolve1d_4D_scan(input, W, mode='full'):
  batch_size, nchannels, nwords, ndim = input.shape
  nkernels_out, nkernels_in, filter_width, ndim = W.shape

  # Unroll filter along columns
  W_unrolled = W.dimshuffle(0, 2, 1, 3).flatten(ndim=3)
  # Replicate input filters 'batch_size' times and squash out_filters along column axis.
  # W_tiled = T.tile(W_unrolled, (1, 1, batch_size)).dimshuffle(1, 0, 2).flatten(ndim=2)  # doesn't give a gradient
  W_tiled = T.alloc(W_unrolled, batch_size, W_unrolled.shape[0], W_unrolled.shape[1], W_unrolled.shape[2]).dimshuffle(1, 2, 0, 3).flatten(ndim=3).dimshuffle(1, 0, 2).flatten(ndim=2)
  W_tiled = W_tiled[::-1]
  # reverse_slicing = [slice(None, None, None)] * W_tiled.ndim
  # reverse_slicing[0] = slice(None, None, -1)
  # reverse_slicing = tuple(reverse_slicing)
  # W_tiled = W_tiled[reverse_slicing]  # flip the kernel

  # Unroll input and pad to fit the output filters.
  input_reshaped = input.dimshuffle(0, 2, 1, 3).flatten(ndim=3).dimshuffle(1,0,2).flatten(ndim=2)
  # input_tiled = T.tile(input_reshaped, (1, nkernels_out))
  input_tiled = T.alloc(input_reshaped, nkernels_out, input_reshaped.shape[0], input_reshaped.shape[1]).dimshuffle(1, 0, 2).flatten(ndim=2)

  if mode == 'full':
    pad = T.zeros((filter_width-1, nkernels_out*batch_size*nchannels*ndim))
    input_padded = T.concatenate([pad, input_tiled, pad])
    conv_out, _ = theano.scan(fn=lambda i: (W_tiled * input_padded[i:i+filter_width]).sum(axis=0),
                              outputs_info=None,
                              sequences=[T.arange(0, nwords+filter_width-1)])
    new_shape = (nwords+filter_width-1, nkernels_out, batch_size, nkernels_in, ndim)
  elif mode == 'valid':
    conv_out, _ = theano.scan(fn=lambda i: (W_tiled * input_tiled[i:i+filter_width]).sum(axis=0),
                              outputs_info=None,
                              sequences=[T.arange(0, nwords-filter_width+1)])
    new_shape = (nwords-filter_width+1, nkernels_out, batch_size, nkernels_in, ndim)

  conv_reshaped = conv_out.reshape(new_shape).dimshuffle(2, 1, 0, 3, 4).sum(axis=3)
  return conv_reshaped


def convolve1d_2D_scan(a, b, mode='full'):
  nwords, ndim = a.shape
  filter_width, ndim = b.shape
  b = b[::-1]
  if mode == 'full':
    pad = T.zeros((filter_width-1, ndim))
    a = T.concatenate([pad, a, pad])
    conv_out, _ = theano.scan(fn=lambda i: (a[i:i+filter_width] * b).sum(axis=0),
                              outputs_info=None,
                              sequences=[T.arange(0, nwords+filter_width-1)])
  elif mode == 'valid':
    conv_out, _ = theano.scan(fn=lambda i: (a[i:i+filter_width], b).sum(axis=0),
                              outputs_info=None,
                              sequences=[T.arange(0, nwords-filter_width+1)])
  return conv_out


def convolve1d_4D_conv2d(input, W, mode='full'):
  conv_out, _ = theano.scan(fn=lambda i: conv2d(input[:,:,:,i:i+1], W[:,:,:,i:i+1], border_mode=mode),
                                outputs_info=None,
                                sequences=[T.arange(0, W.shape[3])])
  conv_out = conv_out.flatten(ndim=4).dimshuffle(1,2,3,0)
  return conv_out


def convolve1d_4D_conv2d_image(input, W, mode='full'):
  return conv2d(input, W, border_mode='valid')


def test_convolve1d_4D(test_grads=False, test_speed=True):
  nbatches, nkernels_in, nwords, ndim = 100, 16, 58, 300
  nkernels_out, filter_width = 4, 7
  # nbatches, nkernels_in, nwords, ndim = 3, 1, 7, 5
  # nkernels_out, filter_width = 2, 3

  input_shape = (nbatches, nkernels_in, nwords, ndim)
  filter_shape = (nkernels_out, nkernels_in, filter_width, ndim)

  image = T.tensor4('input', dtype='float64')
  filt = T.tensor4('filt', dtype='float64')

  # Generate data
  # image_data = np.arange(np.prod(input_shape)).reshape(input_shape)
  image_data = np.random.randn(*input_shape)
  filt_data = np.random.randn(*filter_shape)

  border_mode = 'full'

  # unrolling + einsum
  out_4D = convolve1d_4D(image, filt, mode=border_mode)
  f_conv = theano.function([image, filt], out_4D)

  # using einsum
  out_4D_einsum = convolve1d_4D_einsum(image, filt, mode=border_mode)
  f_conv_einsum = theano.function([image, filt], out_4D_einsum)

  # using theano scan
  out_4D_scan = convolve1d_4D_scan(image, filt, mode=border_mode)
  f_conv_scan = theano.function([image, filt], out_4D_scan)

  # using theano scan and conv2d
  out_4D_conv2d = convolve1d_4D_conv2d(image, filt, mode=border_mode)
  f_conv_conv2d = theano.function([image, filt], out_4D_conv2d)

  out_4D_conv2d_image = convolve1d_4D_conv2d_image(image, filt, mode=border_mode)
  f_conv_conv2d_image = theano.function([image, filt], out_4D_conv2d_image)

  out_conv = f_conv(image_data, filt_data)
  out_conv_einsum = f_conv_einsum(image_data, filt_data)
  out_conv_scan = f_conv_scan(image_data, filt_data)
  out_conv_conv2d = f_conv_conv2d(image_data, filt_data)

  out_conv_conv2d_image = f_conv_conv2d_image(image_data, filt_data)
  print "Checking equality....",
  print list(map(lambda x: x.shape, [out_conv, out_conv_einsum, out_conv_scan, out_conv_conv2d]))
  # assert np.allclose(out_conv, out_conv_einsum, out_conv_scan, out_conv_conv2d)
  print 'out_conv, out_conv_einsum', np.allclose(out_conv, out_conv_einsum)
  print 'out_conv, out_conv_scan', np.allclose(out_conv, out_conv_scan)
  print 'out_conv, out_conv_conv2d', np.allclose(out_conv, out_conv_conv2d)
  print 'out_conv_einsum, out_conv_scan', np.allclose(out_conv_einsum, out_conv_scan)
  print 'out_conv_scan, out_conv_conv2d', np.allclose(out_conv_scan, out_conv_conv2d)
  print "done"

  def check_grads():
    def compute_grad(conv_out):
      rng = T.shared_randomstreams.RandomStreams(seed=234)
      proj = rng.normal(conv_out.shape)
      cost = (conv_out * proj).sum()
      grad = T.grad(cost, [image, filt])
      f_grad = theano.function([image, filt], grad)
      out = f_grad(image_data, filt_data)
      return out

    print 'Comparing gradients...'
    grad_4D = compute_grad(out_4D)
    grad_4D_einsum = compute_grad(out_4D_einsum)
    grad_4D_scan = compute_grad(out_4D_scan)
    grad_4D_conv2d = compute_grad(out_4D_conv2d)
    print 'grad_4D', grad_4D[0].shape, grad_4D[1].shape
    print 'grad_4D_einsum', grad_4D_einsum[0].shape, grad_4D_einsum[1].shape
    print 'grad_4D_scan', grad_4D_scan[0].shape, grad_4D_scan[1].shape
    print 'grad_4D_conv2d', grad_4D_conv2d[0].shape, grad_4D_conv2d[1].shape
    print "Checking equality...."
    # assert np.allclose(grad_4D, grad_4D_einsum, grad_4D_scan)
    print 'grad_4D, grad_4D_einsum', np.allclose(grad_4D[0], grad_4D_einsum[0]), np.allclose(grad_4D[1], grad_4D_einsum[1])
    print 'grad_4D, grad_4D_scan', np.allclose(grad_4D[0], grad_4D_scan[0]), np.allclose(grad_4D[1], grad_4D_scan[1])
    print 'grad_4D_einsum, grad_4D_scan', np.allclose(grad_4D_einsum[0], grad_4D_scan[0]), np.allclose(grad_4D_einsum[1], grad_4D_scan[1])
    print 'grad_4D_einsum, grad_4D_conv2d', np.allclose(grad_4D_einsum[0], grad_4D_conv2d[0]), np.allclose(grad_4D_einsum[1], grad_4D_conv2d[1])
    print "done"

    print "Running unittest_tools.verify_grad...",
    theano.tests.unittest_tools.verify_grad(convolve1d_4D, [image_data, filt_data])
    theano.tests.unittest_tools.verify_grad(convolve1d_4D_einsum, [image_data, filt_data])
    theano.tests.unittest_tools.verify_grad(convolve1d_4D_scan, [image_data, filt_data])
    print "done"

  if test_grads:
    check_grads()

  # Timing
  number = 10
  print 'f_conv', timeit.timeit(lambda : f_conv(image_data, filt_data), number=number)
  print 'f_conv_einsum', timeit.timeit(lambda : f_conv_einsum(image_data, filt_data), number=number)
  print 'f_conv_scan', timeit.timeit(lambda : f_conv_scan(image_data, filt_data), number=number)
  print 'f_conv_scan_conv2d', timeit.timeit(lambda : f_conv_conv2d(image_data, filt_data), number=number)
  print 'f_conv_scan_conv2d_image', timeit.timeit(lambda : f_conv_conv2d_image(image_data, filt_data), number=number)



def test_grad_2d():
  nwords, ndim = 5, 3
  filter_width = 3

  input_shape = (nwords, ndim)
  filter_shape = (filter_width, ndim)

  image = T.matrix('input', dtype='float64')
  filt = T.matrix('filt', dtype='float64')

  # Generate data
  # image_data = np.arange(np.prod(input_shape)).reshape(input_shape)
  rng = np.random.RandomState(123)
  image_data = rng.randn(*input_shape)
  filt_data = rng.randn(*filter_shape)

  border_mode = 'full'
  # unrolling + einsum
  out_2D = convolve1d_2D(image, filt, mode=border_mode)
  f_conv = theano.function([image, filt], out_2D)

  # using theano scan
  out_2D_scan = convolve1d_2D_scan(image, filt, mode=border_mode)
  f_conv_scan = theano.function([image, filt], out_2D_scan)

  ## Compute convo
  out_conv = f_conv(image_data, filt_data)
  out_conv_scan = f_conv_scan(image_data, filt_data)

  assert np.allclose(out_conv, out_conv_scan)

  def compute_grad(conv_out, seed):
    rng = T.shared_randomstreams.RandomStreams(seed=seed)
    proj = rng.normal(out_conv.shape)
    cost = (conv_out * proj).sum()
    grad = T.grad(cost, [image, filt])
    f_grad = theano.function([image, filt], grad)
    out = f_grad(image_data, filt_data)
    return out

  print 'Gradient check'
  for i in xrange(3):
    seed = rng.randint(2**16)
    print i, 'seed=', seed
    grad_2D = compute_grad(out_2D, seed)
    grad_2D_scan = compute_grad(out_2D_scan, seed)
    assert np.allclose(grad_2D[0], grad_2D_scan[0])
    assert np.allclose(grad_2D[1], grad_2D_scan[1])
    # print grad_2D[1]
    # print grad_2D_scan[1]
    # print

  print "Running unittest_tools.verify_grad...",
  theano.tests.unittest_tools.verify_grad(convolve1d_2D_scan, [image_data, filt_data])
  theano.tests.unittest_tools.verify_grad(convolve1d_2D, [image_data, filt_data])
  print "done"



# def kmax_pool(input, k_max):
#   assert input.ndim == 4

#   k = theano.shared(k_max, name='k-max')
#   # Unroll input into 2d ndim x (batch_size x nkernels_in x nwords)
#   pool = TSN.images2neibs(input, (input.shape[2], 1), mode='ignore_borders')
#   neighborsArgSorted = T.argsort(pool, axis=1)
#   yy = T.sort(neighborsArgSorted[:, -k:], axis=1).flatten()
#   xx = T.repeat(T.arange(neighborsArgSorted.shape[0]), k)
#   pool_kmax = pool[xx, yy]
#   pool_kmax_shape = T.join(0, T.as_tensor([input.shape[0], input.shape[1], input.shape[3], k]))
#   pooled_out = pool_kmax.reshape(pool_kmax_shape, ndim=4).dimshuffle(0, 1, 3, 2)
#   return pooled_out


def _k_max_pooling(input, kmax):
  pool = input.dimshuffle(0, 2, 1, 3).flatten(ndim=3).dimshuffle(1,0,2).flatten(ndim=2).dimshuffle(1,0)
  neighborsArgSorted = T.argsort(pool, axis=1)
  yy = T.sort(neighborsArgSorted[:, -kmax:], axis=1).flatten()
  xx = T.repeat(T.arange(neighborsArgSorted.shape[0]), kmax)
  pool_kmax = pool[xx, yy]
  pool_kmax_shape = T.join(0, T.as_tensor([input.shape[0], input.shape[1], input.shape[3], kmax]))
  pooled_out = pool_kmax.reshape(pool_kmax_shape, ndim=4).dimshuffle(0, 1, 3, 2)
  return pooled_out


def k_max_pooling(input, kmax):
  nbatches, nchannels, nwords, ndim = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
  x = input.dimshuffle(0,1,3,2)
  neighborsArgSorted = T.argsort(x, axis=3)
  ax0 = T.repeat(T.arange(nbatches), nchannels*ndim*kmax)
  ax1 = T.repeat(T.arange(nchannels), ndim * kmax).dimshuffle('x', 0)
  ax1 = T.repeat(ax1, nbatches, axis=0).flatten()
  ax2 = T.repeat(T.arange(ndim), kmax, axis=0).dimshuffle('x', 'x', 0)
  ax2 = T.repeat(ax2, nchannels, axis=1)
  ax2 = T.repeat(ax2, nbatches, axis=0).flatten()
  ax3 = T.sort(neighborsArgSorted[:,:,:,-kmax:], axis=3).flatten()

  pooled_out = x[ax0, ax1, ax2, ax3]
  pooled_out = pooled_out.reshape((nbatches, nchannels, ndim, kmax)).dimshuffle(0,1,3,2)
  return pooled_out


def max_pooling(input):
  return T.max(input, axis=2)


def dynamic_k_max_pooling(input, sent_sizes, k_max_factor, k_max_final):
  """
    k_max_factor -- multiplied by sentence_sizes gives the value of kmax for each sentence
  """
  # Unroll input into (batch_size x nchannels x nwords) x ndim
  nbatches, nchannels, nwords, ndim = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
  x = input.dimshuffle(0,1,3,2)

  sent_sizes = T.cast(T.ceil(sent_sizes * k_max_factor), dtype='int32')
  sent_sizes = T.maximum(sent_sizes, k_max_final)
  # sent_sizes_matrix = T.repeat(sent_sizes, nwords, axis=1)
  sent_sizes_matrix = T.repeat(sent_sizes.dimshuffle(0, 'x'), nwords, axis=1)

  idx = T.arange(nwords).dimshuffle('x', 0)
  idx_matrix = T.repeat(idx, nbatches, axis=0)

  sent_sizes_mask = T.lt(idx_matrix, sent_sizes_matrix)[:,::-1]

  neighborsArgSorted = T.argsort(x, axis=3)
  neighborsArgSorted_masked = ((neighborsArgSorted + 1) * sent_sizes_mask.dimshuffle(0,'x','x',1)) - 1
  neighborsArgSorted_masked_sorted = neighborsArgSorted_masked.sort(axis=3)

  nwords_max = T.cast(T.ceil(nwords * k_max_factor), 'int32')
  # print nwords_max.eval()
  neighborsArgSorted_masked_sorted_clipped = neighborsArgSorted_masked_sorted[:,:,:,-nwords_max:]

  ax0 = T.repeat(T.arange(nbatches), nchannels*ndim*nwords_max)
  ax1 = T.repeat(T.arange(nchannels), ndim * nwords_max).dimshuffle('x', 0)
  ax1 = T.repeat(ax1, nbatches, axis=0).flatten()
  ax2 = T.repeat(T.arange(ndim), nwords_max, axis=0).dimshuffle('x', 'x', 0)
  ax2 = T.repeat(ax2, nchannels, axis=1)
  ax2 = T.repeat(ax2, nbatches, axis=0).flatten()
  ax3 = neighborsArgSorted_masked_sorted_clipped.flatten()

  pooled_out = x[ax0, ax1, ax2, ax3]
  pooled_out = pooled_out.reshape((nbatches, nchannels, ndim, nwords_max)).dimshuffle(0,1,3,2)

  return pooled_out


def test_dynamic_k_max_pooling():
  np.random.seed(123)
  nbatches, nkernels_in, nwords, ndim = 3, 1, 58, 20
  input_shape = (nbatches, nkernels_in, nwords, ndim)

  # image_data = np.random.rand(*input_shape)
  data = np.arange(np.prod(input_shape))
  np.random.shuffle(data)
  data = data.reshape(input_shape)
  data[:,:,-2:] = 0.
  print 'data'
  print data
  input = theano.shared(data)

  sent_sizes_data = np.array([3, 5, 20]).astype('int32')#[:,np.newaxis]
  print 'sent_sizes_data'
  print sent_sizes_data
  sent_sizes = theano.shared(sent_sizes_data, borrow=True)

  k_max_factor = 0.5

  pooled_out = dynamic_k_max_pooling(input, sent_sizes, k_max_factor, 2)

  print 'pooled_out'
  print pooled_out.eval().shape
  return


def _max_pooling(input, k):
  return T.sort(input, axis=2)[:,:,-k:,:]


def test_kmax_pool():
  nbatches, nkernels_in, nwords, ndim = 2, 1, 5, 3
  input_shape = (nbatches, nkernels_in, nwords, ndim)

  input = T.tensor4('input')

  k = 3
  f_kmax = theano.function([input], k_max_pooling(input, k))
  f_max = theano.function([input], max_pooling(input))

  image_data = np.arange(np.prod(input_shape), dtype=np.float64)
  np.random.shuffle(image_data)
  image_data = image_data.reshape(input_shape)
  print image_data
  print 'kmax'
  print f_kmax(image_data)
  print 'max'
  print f_max(image_data)


def test_kmax_pooling_time():
  nbatches, nkernels_in, nwords, ndim = 50, 16, 58, 300
  input_shape = (nbatches, nkernels_in, nwords, ndim)

  input = T.tensor4('input')

  k = 1
  f_kmax_argsort = theano.function([input], k_max_pooling(input, k))
  f_kmax_unroll = theano.function([input], _k_max_pooling(input, k))
  f_max = theano.function([input], max_pooling(input))

  image_data = np.random.randn(*input_shape).astype(dtype=np.float64)
  # np.random.shuffle(image_data)
  image_data = image_data.reshape(input_shape)
  # print image_data
  # print 'kmax'
  print 'f_kmax_argsort', timeit.timeit(lambda: f_kmax_argsort(image_data), number=10)
  print 'f_kmax_unroll', timeit.timeit(lambda: f_kmax_unroll(image_data), number=10)
  print 'f_max', timeit.timeit(lambda: f_max(image_data), number=10)


def kmax_pool_unroll():
  pool = input.dimshuffle(0, 2, 1, 3).flatten(ndim=3).dimshuffle(1,0,2).flatten(ndim=2)
  neighborsArgSorted = T.argsort(pool, axis=0)



def test_kmax():
  nbatches, nkernels_in, nwords, ndim = 3, 1, 7, 2
  input_shape = (nbatches, nkernels_in, nwords, ndim)
  image_data = np.ones(input_shape, dtype=np.float64)

  image_data = np.random.rand(*input_shape)
  input = theano.shared(image_data)

  # sent_sizes_data = np.array([3, 2, 3, 2, 4, 5, 3])[:,np.newaxis].astype('int32')
  # sent_sizes = theano.shared(sent_sizes_data, borrow=True)
  # sent_sizes_matrix = T.repeat(sent_sizes, ndim, axis=1)
  # print 'sent_sizes_matrix', sent_sizes_matrix.eval()

  sent_sizes_data = np.random.randint(1, 5, size=(nbatches, 1))
  sent_sizes = theano.shared(sent_sizes_data, borrow=True)
  sent_sizes_matrix = T.repeat(sent_sizes, nwords, axis=1)
  print 'sent_sizes_matrix'
  print sent_sizes_matrix.eval()

  idx = T.arange(nwords).dimshuffle('x', 0)
  idx_matrix = T.repeat(idx, nbatches, axis=0)
  print 'idx_matrix'
  print idx_matrix.eval()

  sent_sizes_mask = T.lt(idx_matrix, sent_sizes_matrix)
  print 'sent_sizes_mask'
  print sent_sizes_mask.eval()

  k_max = 4
  # f_kmax = theano.function([input], kmax_pool(input, k))
  # k = theano.shared(k_max, name='k-max')
  # kmax_limit = nwords * T.ceil(L-l)/L
  # Unroll input into 2d ndim x (batch_size x nkernels_in x nwords)
  # pool = TSN.images2neibs(input, (input.shape[2], 1), mode='ignore_borders')
  print 'input', input.eval()
  neighborsArgSorted = T.argsort(input, axis=2)
  print 'neighborsArgSorted'
  print neighborsArgSorted.eval()
  neighborsArgSorted_masked = (neighborsArgSorted * sent_sizes_mask.dimshuffle(0,'x',1,'x'))

  print 'neighborsArgSorted_masked'
  print neighborsArgSorted_masked.eval()

  neighborsArgSorted_clipped = (neighborsArgSorted * sent_sizes_mask.dimshuffle(0,'x',1,'x'))[:,:,:k_max,:]

  print 'args'
  print neighborsArgSorted_clipped.eval()
  return
  # Given a column of sentence length
  # Tile it along axis=1 to form a matrix
  # Create another matrix with T.arange() to represent indices
  # do T.lt to create a mask and then eliminate all indices in the neighborsArgSorted

  # yy = T.sort(neighborsArgSorted[:, -k:], axis=1).flatten()
  yy = T.sort(neighborsArgSorted_clipped, axis=3).flatten()
  print 'yy', yy.eval()
  xx = T.repeat(T.arange(neighborsArgSorted.shape[0]), k_max)
  pool_kmax = input[xx, yy]

  print pool_kmax.eval()
  # pool_kmax_shape = T.join(0, T.as_tensor([input.shape[0], input.shape[1], input.shape[3], k]))
  # pooled_out = pool_kmax.reshape(pool_kmax_shape, ndim=4).dimshuffle(0, 1, 3, 2)
  pool_kmax_shape = T.join(0, T.as_tensor([input.shape[0], input.shape[1], input.shape[3], kmax_limit]))
  pooled_out = pool_kmax.reshape(pool_kmax_shape, ndim=4).dimshuffle(0, 1, 3, 2)
  # pooled_out = TSN.neibs2images(pool_kmax, (input_shape[2], 1), input_shape, mode='valid') #.dimshuffle(0, 1, 3, 2)

  # image_data = np.arange(np.prod(input_shape), dtype=np.float64).reshape(input_shape)
  print image_data
  print 'kmax', k_max
  # print pooled_out.eval()


def test_convolve1d_4D_conv2d():
  nbatches, nkernels_in, nwords, ndim = 1, 1, 3, 1
  nkernels_out, filter_width = 2, 2

  image_shape = (nbatches, nkernels_in, nwords, ndim)
  filter_shape = (nkernels_out, nkernels_in, filter_width, ndim)

  image = T.tensor4('image', dtype='float64')
  filt = T.tensor4('filt', dtype='float64')

  # Generate data
  # image_data = np.arange(np.prod(image_shape)).reshape(image_shape)
  # filt_data = np.arange(np.prod(filter_shape)).reshape(filter_shape)
  image_data = np.random.randn(*image_shape)
  filt_data = np.random.randn(*filter_shape)

  border_mode = 'full'

  # unrolling + einsum
  out_4D_einsum = convolve1d_4D_einsum(image, filt, mode=border_mode)
  f_conv_einsum = theano.function([image, filt], out_4D_einsum)

  # using theano scan
  out_4D_scan = convolve1d_4D_scan(image, filt, mode=border_mode)
  f_conv_scan = theano.function([image, filt], out_4D_scan)

  # using theano scan and conv2d
  out_4D_conv2d = convolve1d_4D_conv2d(image, filt, mode=border_mode)
  f_conv_conv2d = theano.function([image, filt], out_4D_conv2d)

  out_scan = f_conv_scan(image_data, filt_data)
  out_conv2d = f_conv_conv2d(image_data, filt_data)
  out_einsum = f_conv_einsum(image_data, filt_data)
  assert np.allclose(out_scan, out_conv2d)
  assert np.allclose(out_scan, out_einsum)
  assert np.allclose(out_conv2d, out_einsum)
  # print out_scan
  # print out_conv2d
  # print out_einsum

  def compute_grad(conv_out, seed):
    rng = T.shared_randomstreams.RandomStreams(seed=seed)
    proj = rng.normal(conv_out.shape)
    cost = (conv_out * proj).sum()
    grad = T.grad(cost, [image, filt])
    f_grad = theano.function([image, filt], grad)
    out = f_grad(image_data, filt_data)
    return out

  print 'Gradient check'
  rng = np.random.RandomState(123)
  for i in xrange(3):
    seed = rng.randint(2**16)
    print i, 'seed=', seed
    grad_scan = compute_grad(out_4D_scan, seed)
    grad_einsum = compute_grad(out_4D_einsum, seed)
    assert np.allclose(grad_einsum[0], grad_scan[0])
    assert np.allclose(grad_einsum[1], grad_scan[1])

  print 'convolve1d_4D_einsum'
  theano.tests.unittest_tools.verify_grad(convolve1d_4D_einsum, [image_data, filt_data])


if __name__ == '__main__':
  np.random.seed(232)
  # test_convolve1d_4D()
  # test_grad_2d()
  # test_kmax()
  test_kmax_pool()
  # test_kmax_pooling_time()
  # test_convolve1d_4D_conv2d()
  # test_dynamic_k_max_pooling()
