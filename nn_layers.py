import cPickle
import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.sandbox.neighbours as TSN
from theano.tensor.shared_randomstreams import RandomStreams

import conv1d


def load_nnet(nnet_fname, params_fname=None):
  print "Loading Nnet topology from", nnet_fname
  train_nnet, test_nnet = cPickle.load(open(nnet_fname, 'rb'))
  if params_fname:
    print "Loading network params from", params_fname
    params = train_nnet.params
    best_params = cPickle.load(open(params_fname, 'rb'))
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)

  for p_train, p_test in zip(train_nnet.params[:-2], test_nnet.params[:-2]):
    assert p_train == p_test
    assert numpy.allclose(p_train.get_value(), p_test.get_value())

  return train_nnet, test_nnet


def relu(x):
    return T.maximum(0.0, x)

def relu_f(z):
    """ Wrapper to quickly change the rectified linear unit function """
    return z * (z > 0.)


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape, dtype=theano.config.floatX)
        return x * mask
    return x


class Layer(object):
  def __init__(self):
    self.params = []
    self.weights = []
    self.biases = []

  def output_func(self, input):
    raise NotImplementedError("Each concrete class needs to implement output_func")

  def set_input(self, input):
    self.output = self.output_func(input)

  def __repr__(self):
    return "{}".format(self.__class__.__name__)


class FeedForwardNet(Layer):
  def __init__(self, layers=None, name=None):
    super(FeedForwardNet, self).__init__()

    if not name:
      name = self.__class__.__name__
    self.name = name

    self.layers = layers if layers else []
    for layer in layers:
      self.weights.extend(layer.weights)
      self.biases.extend(layer.biases)
      self.params.extend(layer.weights + layer.biases)
    self.num_params = sum([numpy.prod(p.shape.eval()) for p in self.params])

  def output_func(self, input):
    cur_input = input
    for layer in self.layers:
      layer.set_input(cur_input)
      cur_input = layer.output
    return cur_input

  def __repr__(self):
    layers_str = '\n'.join(['\t{}'.format(line) for layer in self.layers for line in str(layer).splitlines()])
    return '{} [num params: {}]\n{}'.format(self.name, self.num_params, layers_str)


class ParallelLayer(FeedForwardNet):

  def output_func(self, input):
    layers_out = []
    for layer in self.layers:
      layer.set_input(input)
      layers_out.append(layer.output)
    return T.concatenate(layers_out, axis=1)


class DropoutLayer(Layer):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng, p=0.5):
      super(DropoutLayer, self).__init__()
      seed = rng.randint(2 ** 30)
      self.srng = RandomStreams(seed)
      self.p = p

    def output_func(self, input):
      mask = self.srng.binomial(n=1, p=1.-self.p, size=input.shape, dtype=theano.config.floatX)
      return input * mask

    def __repr__(self):
      return "{}: p={}".format(self.__class__.__name__, self.p)


class FastDropoutLayer(Layer):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng):
      super(FastDropoutLayer, self).__init__()
      seed = rng.randint(2 ** 30)
      self.srng = RandomStreams(seed)

    def output_func(self, input):
      mask = self.srng.normal(size=input.shape, avg=1., dtype=theano.config.floatX)
      return input * mask

    def __repr__(self):
      return "{}".format(self.__class__.__name__)


class LookupTable(Layer):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, W=None):
      super(LookupTable, self).__init__()
      self.W = theano.shared(value=W, name='W_emb', borrow=True)
      self.weights = [self.W]

    def output_func(self, input):
      return self.W[input]

    def __repr__(self):
      return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())


class LookupTableFast(Layer):
    """ Basic linear transformation layer (W.X + b).
    Padding is used to force conv2d with valid mode behave as working in full mode."""
    def __init__(self, W=None, pad=None):
      super(LookupTableFast, self).__init__()
      self.pad = pad
      self.W = theano.shared(value=W, name='W_emb', borrow=True)
      self.weights = [self.W]

    def output_func(self, input):
      out = self.W[input.flatten()].reshape((input.shape[0], 1, input.shape[1], self.W.shape[1]))
      if self.pad:
        pad_matrix = T.zeros((out.shape[0], out.shape[1], self.pad, out.shape[3]))
        out = T.concatenate([pad_matrix, out, pad_matrix], axis=2)
      return out

    def __repr__(self):
      return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())


class PadLayer(Layer):
  def __init__(self, pad, axis=2):
      super(PadLayer, self).__init__()
      self.pad = pad
      self.axis = axis

  def output_func(self, input):
    pad_matrix = T.zeros((input.shape[0], input.shape[1], self.pad, input.shape[3]))
    out = T.concatenate([pad_matrix, input, pad_matrix], axis=self.axis)
    return out



class LookupTableFastStatic(Layer):
    """ Basic linear transformation layer (W.X + b).
    Padding is used to force conv2d with valid mode behave as working in full mode."""
    def __init__(self, W=None, pad=None):
      super(LookupTableFastStatic, self).__init__()
      self.pad = pad
      self.W = theano.shared(value=W, name='W_emb', borrow=True)

    def output_func(self, input):
      out = self.W[input.flatten()].reshape((input.shape[0], 1, input.shape[1], self.W.shape[1]))
      if self.pad:
        pad_matrix = T.zeros((out.shape[0], out.shape[1], self.pad, out.shape[3]))
        out = T.concatenate([pad_matrix, out, pad_matrix], axis=2)
      return out

    def __repr__(self):
      return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())


class LookupTableStatic(LookupTable):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, W=None):
      super(LookupTableStatic, self).__init__(W)
      self.weights = []


class ParallelLookupTable(FeedForwardNet):
  def output_func(self, x):
    layers_out = []
    assert len(x) == len(self.layers)
    for x, layer in zip(x, self.layers):
      layer.set_input(x)
      layers_out.append(layer.output)
    return T.concatenate(layers_out, axis=3)


class FlattenLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """

  def output_func(self, input):
    return input.flatten(2)


class MaxOutLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """
  def __init__(self, maxout_size):
    super(MaxOutLayer, self).__init__()
    self.maxout_size = maxout_size

  def output_func(self, input):
    # MaxOut across feature maps (channels)
    maxout_out = None
    for i in xrange(self.maxout_size):
      t = input[:,i::self.maxout_size,:,:]
      if maxout_out is None:
        maxout_out = t
      else:
        maxout_out = T.maximum(maxout_out, t)
    return maxout_out


class MaxOutFoldingLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """
  def __init__(self, maxout_size):
    super(MaxOutFoldingLayer, self).__init__()
    self.maxout_size = maxout_size

  def output_func(self, input):
    # MaxOut across feature maps (channels)
    maxout_out = None
    for i in xrange(self.maxout_size):
      t = input[:,:,:,i::self.maxout_size]
      if maxout_out is None:
        maxout_out = t
      else:
        maxout_out = T.maximum(maxout_out, t)

    return maxout_out

  def __repr__(self):
    return "{}: maxout_size={}".format(self.__class__.__name__, self.maxout_size)


class TranformInputTo4DTensorLayer(Layer):

  def output_func(self, input):
   return input.dimshuffle((0, 'x', 1, 2))


class LinearLayer(Layer):
  def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh):
    super(LinearLayer, self).__init__()

    if W is None:
      W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)

      W = theano.shared(value=W_values, name='W', borrow=True)
    if b is None:
      b = build_shared_zeros((n_out,), 'b')

    self.W = W
    self.b = b

    self.activation = activation

    self.weights = [self.W]
    self.biases = [self.b]

  def output_func(self, input):
    return self.activation(T.dot(input, self.W) + self.b)

  def __repr__(self):
    return "{}: W_shape={} b_shape={} activation={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval(), self.activation)



class NonLinearityLayer(Layer):
  def __init__(self, b_size, b=None, activation=T.tanh):
    super(NonLinearityLayer, self).__init__()
    if not b:
      b_values = numpy.zeros(b_size, dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)
    self.b = b
    self.activation = activation
    # In input we get a tensor (batch_size, nwords, ndim)
    self.biases = [self.b]

  def output_func(self, input):
    return self.activation(input + self.b.dimshuffle('x', 0, 'x', 'x'))

  def __repr__(self):
    return "{}: b_shape={} activation={}".format(self.__class__.__name__, self.b.shape.eval(), self.activation)


class NonLinearityLayerForConv1d(Layer):

  def __init__(self, b_shape, b=None, activation=T.tanh, rng=None):
    super(NonLinearityLayerForConv1d, self).__init__()
    self.rng = rng
    self.activation = activation
    if not b:
      if rng:
        b_values = rng.uniform(size=b_shape).astype(dtype=theano.config.floatX)
      else:
        b_values = numpy.zeros(b_shape, dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)
    self.b = b
    self.biases = [self.b]

  def output_func(self, input):
    return self.activation(input + self.b.dimshuffle('x', 0, 'x', 1))

  def __repr__(self):
    return "{}: shape={} activation={} rng={}".format(self.__class__.__name__, self.b.shape.eval(), self.activation, self.rng)


class FoldingLayer(Layer):
  """Folds across last axis (ndim)."""

  def output_func(self, input):
    # assert (input.shape[3].eval() % 2 == 0)
    # In input we get a tensor (batch_size, nwords, ndim)
    return input[:,:,:,::2] + input[:,:,:,1::2]
    # return input[:,:,:,:-1:2] + input[:,:,:,1::2]


class FoldingLayerSym(Layer):
  """Folds across last axis (ndim)."""

  def output_func(self, input):
    # In input we get a tensor (batch_size, nwords, ndim)
    mid = input.shape[3] // 2
    return input[:,:,:,:mid] + input[:,:,:,mid:]


class KMaxPoolLayer(Layer):
  """Folds across last axis (ndim)."""
  def __init__(self, k_max):
    super(KMaxPoolLayer, self).__init__()
    self.k_max = k_max

  def output_func(self, input):
    # In input we get a tensor (batch_size, nwords, ndim)
    if self.k_max == 1:
      return conv1d.max_pooling(input)
    return conv1d.k_max_pooling(input, self.k_max)

  def __repr__(self):
    return "{}: k_max={}".format(self.__class__.__name__, self.k_max)


class MaxPoolLayer(Layer):
  """Folds across last axis (ndim)."""
  def __init__(self, pool_size):
    super(MaxPoolLayer, self).__init__()
    self.pool_size = pool_size

  def output_func(self, input):
    # In input we get a tensor (batch_size, nwords, ndim)
    return downsample.max_pool_2d(input=input, ds=self.pool_size, ignore_border=True)

  def __repr__(self):
    return "{}: pool_size={}".format(self.__class__.__name__, self.pool_size)


class ConvolutionLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """
  def __init__(self, rng, filter_shape, input_shape=None, W=None):
    super(ConvolutionLayer, self).__init__()
    # initialize weights with random weights
    if W is None:
      # there are "num input feature maps * filter height * filter width"
      # inputs to each hidden unit
      fan_in = numpy.prod(filter_shape[1:])
      # each unit in the lower layer receives a gradient from:
      # "num output feature maps * filter height * filter width" /
      #   pooling size
      fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
      W_bound = numpy.sqrt(1. / fan_in)
      # W_bound = numpy.sqrt(6. / (fan_in + fan_out))
      W_data = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)

      # Simple initialization
      # W_data = 0.05 * rng.randn(*filter_shape).astype(dtype=theano.config.floatX)
      W = theano.shared(W_data, name="W_conv1d", borrow=True)

    self.filter_shape = filter_shape
    self.input_shape = input_shape

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: filter_shape={}; input_shape={}".format(self.__class__.__name__, self.W.shape.eval(), self.input_shape)


class Conv1dLayer(ConvolutionLayer):

  def output_func(self, input):
    return conv1d.convolve1d_4D(input, self.W, mode='full')


class Conv2dLayer(ConvolutionLayer):

  def output_func(self, input):
    return conv.conv2d(input, self.W, border_mode='valid',
                       filter_shape=self.filter_shape,
                       image_shape=self.input_shape)


# def Conv2dMaxPool(rng, filter_shape, activation):
#   conv = Conv2dLayer(rng, filter_shape)
#   nonlinearity = NonLinearityLayer(activation=activation)
#   pooling = MaxPoolLayer()
#   layer = FeedForwardNet(layers=[])
#   return layer


class LogisticRegression(Layer):
    """Multi-class Logistic Regression
    """
    def __init__(self, n_in, n_out, W=None, b=None):
      if not W:
        W = build_shared_zeros((n_in, n_out), 'W_softmax')
      if not b:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.b = b
      self.weights = [self.W]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        self.p_y_given_x = T.nnet.softmax(self._dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred

    def _dot(self, a, b):
        return T.dot(a, b)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood(y)

    def training_cost_weighted(self, y, weights=None):
        """ Wrapper for standard name """
        LL = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        weights = T.repeat(weights.dimshuffle('x', 0), y.shape[0], axis=0)
        factors = weights[T.arange(y.shape[0]), y]
        return -T.mean(LL * factors)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def f1_score(self, y, labels=[0, 2]):
      """
      Mean F1 score between two classes (positive and negative as specified by the labels array).
      """
      y_tr = y
      y_pr = self.y_pred

      correct = T.eq(y_tr, y_pr)
      wrong = T.neq(y_tr, y_pr)

      label = labels[0]
      tp_neg = T.sum(correct * T.eq(y_tr, label))
      fp_neg = T.sum(wrong * T.eq(y_pr, label))
      fn_neg = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_neg = T.cast(tp_neg, theano.config.floatX)
      prec_neg = tp_neg / T.maximum(1, tp_neg + fp_neg)
      recall_neg = tp_neg / T.maximum(1, tp_neg + fn_neg)
      f1_neg = 2. * prec_neg * recall_neg / T.maximum(1, prec_neg + recall_neg)

      label = labels[1]
      tp_pos = T.sum(correct * T.eq(y_tr, label))
      fp_pos = T.sum(wrong * T.eq(y_pr, label))
      fn_pos = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_pos = T.cast(tp_pos, theano.config.floatX)
      prec_pos = tp_pos / T.maximum(1, tp_pos + fp_pos)
      recall_pos = tp_pos / T.maximum(1, tp_pos + fn_pos)
      f1_pos = 2. * prec_pos * recall_pos / T.maximum(1, prec_pos + recall_pos)

      return 0.5 * (f1_pos + f1_neg) * 100

    def accuracy(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        return T.mean(T.eq(self.y_pred, y)) * 100



class L2SVM(Layer):
    """Multi-class Logistic Regression
    """
    def __init__(self, n_in, n_out, W=None, b=None):
      if not W:
        W = build_shared_zeros((n_in, n_out), 'W_softmax')
      if not b:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.b = b
      self.weights = [self.W]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        self.f = self._dot(input, self.W) + self.b
        self.y_pred = T.argmax(self.f, axis=1)
        return self.y_pred

    def _dot(self, a, b):
        return T.dot(a, b)

    def hinge_sq(self, y):
      hinge = T.maximum(1. - self.f[T.arange(y.shape[0]), self.y_pred] * (-1)**T.neq(y, self.y_pred), 0.)
      return hinge**2

    def training_cost(self, y):
        """ Wrapper for standard name """
        return T.sum(self.hinge_sq(y))
        # return T.mean(hinge**2)

    def training_cost_weighted(self, y, weights=None):
        """ Wrapper for standard name """
        loss = self.hinge_sq(y)
        weights = T.repeat(weights.dimshuffle('x', 0), y.shape[0], axis=0)
        factors = weights[T.arange(y.shape[0]), y]
        return T.sum(loss * factors)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def f1_score(self, y, labels=[0, 2]):
      """
      Mean F1 score between two classes (positive and negative as specified by the labels array).
      """
      y_tr = y
      y_pr = self.y_pred

      correct = T.eq(y_tr, y_pr)
      wrong = T.neq(y_tr, y_pr)

      label = labels[0]
      tp_neg = T.sum(correct * T.eq(y_tr, label))
      fp_neg = T.sum(wrong * T.eq(y_pr, label))
      fn_neg = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_neg = T.cast(tp_neg, theano.config.floatX)
      prec_neg = tp_neg / T.maximum(1, tp_neg + fp_neg)
      recall_neg = tp_neg / T.maximum(1, tp_neg + fn_neg)
      f1_neg = 2. * prec_neg * recall_neg / T.maximum(1, prec_neg + recall_neg)

      label = labels[1]
      tp_pos = T.sum(correct * T.eq(y_tr, label))
      fp_pos = T.sum(wrong * T.eq(y_pr, label))
      fn_pos = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_pos = T.cast(tp_pos, theano.config.floatX)
      prec_pos = tp_pos / T.maximum(1, tp_pos + fp_pos)
      recall_pos = tp_pos / T.maximum(1, tp_pos + fn_pos)
      f1_pos = 2. * prec_pos * recall_pos / T.maximum(1, prec_pos + recall_pos)

      return 0.5 * (f1_pos + f1_neg) * 100

    def accuracy(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        return T.mean(T.eq(self.y_pred, y)) * 100


class PairwiseLogisticRegression(Layer):
    """Multi-class Logistic Regression
    """
    def __init__(self, q_in, a_in, n_out, W=None, b=None):
      if not W:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax')
      if not b:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.b = b
      self.weights = [self.W]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a = input[0], input[1]
        dot = T.batched_dot(q, T.dot(a, self.W))

        self.p_y_given_x = T.nnet.softmax(dot + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred

    def _dot(self, a, b):
        return T.dot(a, b)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood(y)

    def training_cost_weighted(self, y, weights=None):
        """ Wrapper for standard name """
        LL = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        weights = T.repeat(weights.dimshuffle('x', 0), y.shape[0], axis=0)
        factors = weights[T.arange(y.shape[0]), y]
        return -T.mean(LL * factors)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def accuracy(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        return T.mean(T.eq(self.y_pred, y)) * 100


class PairwiseLogisticWithFeatsRegression(PairwiseLogisticRegression):
    def __init__(self, q_in, a_in, n_in, n_out, W=None, W_feats=None, b=None):
      if W is None:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax_pairwise')
      if W_feats is None:
        W_feats = build_shared_zeros((n_in, n_out), 'W_softmax_feats')
      if b is None:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.W_feats = W_feats
      self.b = b

      self.lamda = build_shared_zeros((n_out,), 'lamda')

      self.weights = [self.W, self.W_feats]
      self.biases = [self.b, self.lamda]


    def __repr__(self):
      return "{}: W={}, W_feats={}, b={}, lambda".format(self.__class__.__name__, self.W.shape.eval(), self.W_feats.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a, feats = input[0], input[1], input[2]

        dot = T.batched_dot(q, T.dot(a, self.W))
        feats_dot = T.dot(feats, self.W_feats)
        l = self.lamda.dimshuffle('x', 0)
        self.p_y_given_x = T.nnet.softmax(l*dot + (1-l) * feats_dot + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred


class PairwiseLogisticOnlySimRegression(PairwiseLogisticRegression):
    def __init__(self, q_in, a_in, n_out, W=None, b=None):
      if W is None:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax_pairwise')
      if b is None:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.b = b

      self.weights = [self.W]
      self.biases = [self.b]


    def __repr__(self):
      return "{}: W={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a = input[0], input[1]
        # dot = T.batched_dot(q, T.dot(a, self.W.T))
        dot = T.batched_dot(q, T.dot(a, self.W))
        self.p_y_given_x = T.nnet.softmax(dot + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred


class __PairwiseLogisticWithFeatsRegression(PairwiseLogisticRegression):
    def __init__(self, q_in, a_in, n_in, n_out, W=None, W_feats=None, b=None):
      if W is None:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax_pairwise')
      if W_feats is None:
        W_feats = build_shared_zeros((n_in, n_out), 'W_softmax_feats')
      if b is None:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.W_feats = W_feats
      self.b = b

      self.weights = [self.W, self.W_feats]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, W_feats={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.W_feats.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a, feats = input[0], input[1], input[2]

        dot = T.batched_dot(q, T.dot(a, self.W))
        feats_dot = T.dot(feats, self.W_feats)

        self.p_y_given_x = T.nnet.softmax(dot + feats_dot + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred


class PairwiseWithFeatsLayer(Layer):
  def __init__(self, q_in, a_in, n_in, activation=T.tanh):
    super(PairwiseWithFeatsLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a, feats = input[0], input[1], input[2]

      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      dot = T.batched_dot(q, T.dot(a, self.W.T))
      out = T.concatenate([dot.dimshuffle(0, 'x'), q, a, feats], axis=1)
      return out


class PairwiseNoFeatsLayer(Layer):
  def __init__(self, q_in, a_in, activation=T.tanh):
    super(PairwiseNoFeatsLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a = input[0], input[1]
      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      dot = T.batched_dot(q, T.dot(a, self.W.T))
      out = T.concatenate([dot.dimshuffle(0, 'x'), q, a], axis=1)
      return out


class PairwiseOnlySimWithFeatsLayer(Layer):
  def __init__(self, q_in, a_in, n_in, activation=T.tanh):
    super(PairwiseOnlySimWithFeatsLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a, feats = input[0], input[1], input[2]

      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      dot = T.batched_dot(q, T.dot(a, self.W.T))
      out = T.concatenate([dot.dimshuffle(0, 'x'), feats], axis=1)
      # out = feats
      return out


class PairwiseLayer(Layer):
  def __init__(self, q_in, a_in, activation=T.tanh):
    super(PairwiseLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a = input[0], input[1]

      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      dot = T.batched_dot(q, T.dot(a, self.W.T))
      out = T.concatenate([dot.dimshuffle(0, 'x'), q, a], axis=1)
      return out


class PairwiseLayerMulti(Layer):
  def __init__(self, q_in, a_in, activation=T.tanh):
    super(PairwiseLayerMulti, self).__init__()
    ndim = 50
    self.Wq = build_shared_zeros((q_in, ndim), 'W_softmax_pairwise')
    self.Wa = build_shared_zeros((a_in, ndim), 'W_softmax_pairwise')

    self.weights = [self.Wq, self.Wa]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a = input[0], input[1]

      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      qdot = T.dot(q, self.Wq)
      adot = T.dot(a, self.Wa)
      dot = T.batched_dot(qdot, adot)
      out = T.concatenate([dot.dimshuffle(0, 'x'), q, a], axis=1)
      return out


class PairwiseMultiOnlySimWithFeatsLayer(Layer):
  def __init__(self, q_in, a_in, n_in, activation=T.tanh):
    super(PairwiseMultiOnlySimWithFeatsLayer, self).__init__()
    ndim = 50
    self.Wq = build_shared_zeros((q_in, ndim), 'W_softmax_pairwise')
    self.Wa = build_shared_zeros((a_in, ndim), 'W_softmax_pairwise')
    self.weights = [self.Wq, self.Wa]

  def __repr__(self):
    return "{}: {}".format(self.__class__.__name__,
                           ' '.join(['W={}'.format(w.shape.eval()) for w in self.weights]))

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a, feats = input[0], input[1], input[2]

      qdot = T.dot(q, self.Wq)
      adot = T.dot(a, self.Wa)
      dot = qdot * adot
      out = T.concatenate([dot, feats, q, a], axis=1)
      return out


class PairwiseOnlySimScoreLayer(Layer):
  def __init__(self, q_in, a_in, activation=T.tanh):
    super(PairwiseOnlySimScoreLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a = input[0], input[1]

      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      out = T.batched_dot(q, T.dot(a, self.W.T)).dimshuffle(0, 'x')
      return out


class _PairwiseLogisticWithFeatsRegression(PairwiseLogisticRegression):
    def __init__(self, q_in, a_in, n_in, n_out, W=None, W_feats=None, b=None):
      if W is None:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax_pairwise')
        W_q = build_shared_zeros((q_in, n_out), 'W_softmax_q')
        W_a = build_shared_zeros((a_in, n_out), 'W_softmax_a')
      if W_feats is None:
        W_feats = build_shared_zeros((n_in, n_out), 'W_softmax_feats')
      if b is None:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.W_q = W_q
      self.W_a = W_a
      self.W_feats = W_feats
      self.b = b

      self.weights = [self.W, self.W_feats, self.W_q, self.W_a]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W_pairwise={}, W_q={}, W_a={}, W_feats={}, b={}".format(self.__class__.__name__,
                                                                          self.W.shape.eval(),
                                                                          self.W_q.shape.eval(),
                                                                          self.W_a.shape.eval(),
                                                                          self.W_feats.shape.eval(),
                                                                          self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a, feats = input[0], input[1], input[2]

        dot = T.batched_dot(q, T.dot(a, self.W))
        feats_dot = T.dot(feats, self.W_feats)

        self.p_y_given_x = T.nnet.softmax(dot + feats_dot + T.dot(q, self.W_q) + T.dot(a, self.W_a) + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred


class PairwiseL2SVMWithFeatsRegression(PairwiseLogisticRegression):
    def __init__(self, q_in, a_in, n_in, n_out, W=None, W_feats=None, b=None):
      if W is None:
        W = build_shared_zeros((q_in, a_in, n_out), 'W_softmax_pairwise')
      if W_feats is None:
        W_feats = build_shared_zeros((n_in, n_out), 'W_softmax_feats')
      if b is None:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.W_feats = W_feats
      self.b = b

      self.weights = [self.W, self.W_feats]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, W_feats={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.W_feats.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # P(Y|X) = softmax(W.X + b)
        q, a, feats = input[0], input[1], input[2]

        dot = T.batched_dot(q, T.dot(a, self.W))
        feats_dot = T.dot(feats, self.W_feats)

        self.p_y_given_x = T.nnet.softmax(dot + feats_dot + self.b.dimshuffle('x', 0))
        self.prob = self.p_y_given_x[:,-1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred

    def _dot(self, a, b):
        return T.dot(a, b)

    def hinge_sq(self, y):
      hinge = T.maximum(1. - self.p_y_given_x[T.arange(y.shape[0]), self.y_pred] * (-1)**T.neq(y, self.y_pred), 0.)
      return hinge**2

    def training_cost(self, y):
        """ Wrapper for standard name """
        return T.sum(self.hinge_sq(y))
        # return T.mean(hinge**2)

    def training_cost_weighted(self, y, weights=None):
        """ Wrapper for standard name """
        loss = self.hinge_sq(y)
        weights = T.repeat(weights.dimshuffle('x', 0), y.shape[0], axis=0)
        factors = weights[T.arange(y.shape[0]), y]
        return T.sum(loss * factors)