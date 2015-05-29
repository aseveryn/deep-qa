import numpy
import theano
from theano import tensor as T
from collections import OrderedDict
import time
from fish import ProgressFish
from nn_layers import build_shared_zeros



class MiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, rng, datasets, batch_size=100, randomize=False):
        self.rng = rng
        self.datasets = datasets
        self.batch_size = batch_size
        self.n_samples = self.datasets[0].shape[0]
        self.n_batches = (self.n_samples + self.batch_size - 1) / self.batch_size
        # self.n_batches = self.n_samples / self.batch_size  # Prevents the last batch to be smaller than batch_size (this makes conv2d fail)
        self.randomize = randomize

    def __len__(self):
      return self.n_batches

    def __iter__(self):
        n_batches = self.n_batches
        batch_size = self.batch_size
        n_samples = self.n_samples
        if self.randomize:
            # for _ in xrange(self.n_samples / self.batch_size):
            for _ in xrange(n_batches):
                if batch_size > 1:
                    i = int(self.rng.rand(1) * n_batches)
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield [x[i*batch_size:min((i+1)*batch_size, n_samples)] for x in self.datasets]
        else:
            for i in xrange(n_batches):
                yield [x[i*batch_size:min((i+1)*batch_size, n_samples)] for x in self.datasets]


class MiniBatchIteratorConstantBatchSize(object):
    """ Basic mini-batch iterator """
    def __init__(self, rng, datasets, batch_size=100, randomize=False):
        self.rng = rng

        self.batch_size = batch_size
        self.n_samples = datasets[0].shape[0]
        padded_datasets = []
        for d in datasets:
          pad_size = batch_size - len(d) % batch_size
          pad = d[:pad_size]
          # print 'd.shape, pad', d.shape, pad.shape
          padded_dataset = numpy.concatenate([d, pad])
          padded_datasets.append(padded_dataset)
        self.datasets = padded_datasets
        self.n_batches = (self.n_samples + self.batch_size - 1) / self.batch_size
        # self.n_batches = self.n_samples / self.batch_size

        self.randomize = randomize
        # print 'n_samples', self.n_samples
        # print 'n_batches', self.n_batches

    def __len__(self):
      return self.n_batches

    def __iter__(self):
        n_batches = self.n_batches
        batch_size = self.batch_size
        n_samples = self.n_samples
        if self.randomize:
            for _ in xrange(n_batches):
              i = self.rng.randint(n_batches)
              yield [x[i*batch_size:(i+1)*batch_size] for x in self.datasets]
        else:
            for i in xrange(n_batches):
              yield [x[i*batch_size:(i+1)*batch_size] for x in self.datasets]

class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, rng, x, y, batch_size=100, randomize=False):
        self.rng = rng
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n_samples = self.x.shape[0]
        self.n_batches = (self.n_samples + self.batch_size - 1) / self.batch_size
        # self.n_batches = self.n_samples / self.batch_size  # Prevents the last batch to be smaller than batch_size (this makes conv2d fail)
        self.randomize = randomize

    def __len__(self):
      return self.n_batches

    def __iter__(self):
        if self.randomize:
            # for _ in xrange(self.n_samples / self.batch_size):
            for _ in xrange(self.n_batches):
                if self.batch_size > 1:
                    i = int(self.rng.rand(1) * self.n_batches)
                else:
                    i = int(math.floor(self.rng.rand(1) * self.n_samples))
                yield (self.x[i*self.batch_size:min((i+1)*self.batch_size, self.n_samples)],
                       self.y[i*self.batch_size:min((i+1)*self.batch_size, self.n_samples)])
        else:
            for i in xrange(self.n_batches):
                yield (self.x[i*self.batch_size:min((i+1)*self.batch_size, self.n_samples)],
                       self.y[i*self.batch_size:min((i+1)*self.batch_size, self.n_samples)])



def get_sgd_updates(cost, params, learning_rate=0.1, max_norm=9, rho=0.95):
    """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
    """
    print "Generating sgd updates"
    gparams = T.grad(cost, params)

    # compute list of weights updates
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        if max_norm:
            W = param - gparam * learning_rate
            col_norms = W.norm(2, axis=0)
            desired_norms = T.clip(col_norms, 0, max_norm)
            updates[param] = W * (desired_norms / (1e-6 + col_norms))
        else:
            updates[param] = param - gparam * learning_rate
    return updates


def get_adagrad_updates(mean_cost, params, learning_rate=0.1, max_norm=9, _eps=1e-6):
    """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
    """
    print "Generating adagrad updates"
    # compute the gradients with respect to the model parameters
    gparams = T.grad(mean_cost, params)

    accugrads = []
    for param in params:
      accugrads.append(build_shared_zeros(param.shape.eval(), 'accugrad'))

    # compute list of weights updates
    updates = OrderedDict()
    for accugrad, param, gparam in zip(accugrads, params, gparams):
        # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
        agrad = accugrad + gparam * gparam
        dx = - (learning_rate / T.sqrt(agrad + _eps)) * gparam
        update = param + dx
        if max_norm:
            W = param + dx
            col_norms = W.norm(2, axis=0)
            desired_norms = T.clip(col_norms, 0, max_norm)
            update = W * (desired_norms / (1e-6 + col_norms))

        updates[param] = update
        updates[accugrad] = agrad
    return updates


def get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=9, word_vec_name='W_emb'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    print "Generating adadelta updates"
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        exp_sqr_grads[param] = build_shared_zeros(param.shape.eval(), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = build_shared_zeros(param.shape.eval(), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + eps) / T.sqrt(up_exp_sg + eps)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        # if (param.get_value(borrow=True).ndim == 2) and (param.name != word_vec_name):
        if max_norm and param.name != word_vec_name:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(max_norm))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def _get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=9, word_vec_name='W_emb'):
  print "Generating adadelta updates (implementation from dnn)"
  # compute list of weights updates
  gparams = T.grad(cost, params)

  accugrads, accudeltas = [], []
  for param in params:
    accugrads.append(build_shared_zeros(param.shape.eval(), 'accugrad'))
    accudeltas.append(build_shared_zeros(param.shape.eval(), 'accudelta'))

  # compute list of weights updates
  updates = OrderedDict()

  for accugrad, accudelta, param, gparam in zip(accugrads, accudeltas, params, gparams):
      # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
      agrad = rho * accugrad + (1 - rho) * gparam * gparam
      dx = - T.sqrt((accudelta + eps) / (agrad + eps)) * gparam
      updates[accudelta] = (rho * accudelta + (1 - rho) * dx * dx)
      if (max_norm > 0) and param.ndim == 2 and param.name != word_vec_name:
          W = param + dx
          col_norms = W.norm(2, axis=0)
          desired_norms = T.clip(col_norms, 0, T.sqrt(max_norm))
          updates[param] = W * (desired_norms / (1e-7 + col_norms))
      else:
          updates[param] = param + dx
      updates[accugrad] = agrad
  return updates


class Trainer(object):
  def __init__(self, rng, cost, errors, params, method, learning_rate=0.01, max_norm=9):
    self.rng = rng
    self.cost = cost
    self.errors = errors
    self.params = params

    self.batch_x = T.lmatrix('batch_x')
    self.batch_y = T.ivector('batch_y')

    if method == 'adagrad':
      self.updates = get_adagrad_updates(cost, params, learning_rate=learning_rate, max_norm=max_norm, _eps=1e-6)


  def _batch_score(self, batch_iterator):
    """ returned function that scans the entire set given as input """
    score_fn = theano.function(inputs=[self.batch_x, self.batch_y],
                               outputs=self.errors,
                               givens={x: batch_x, y: batch_y})
    def foo():
      return [score_fn(batch_x, batch_y) for batch_x, batch_y in batch_iterator]
    return foo

  def fit(self, x_train, y_train, x_dev=None, y_dev=None, batch_size=100):
    train_fn = theano.function(inputs=[self.batch_x, self.batch_y],
                               outputs=self.cost,
                               updates=self.updates,
                               givens={x: self.batch_x, y: self.batch_y})

    train_set_iterator = DatasetMiniBatchIterator(self.rng, x_train, y_train, batch_size=batch_size, randomize=True)
    dev_set_iterator = DatasetMiniBatchIterator(self.rng, x_dev, y_dev, batch_size=batch_size, randomize=False)

    train_score = self._batch_score(train_set_iterator)
    dev_score = self._batch_score(dev_set_iterator)

    best_dev_error = numpy.inf
    epoch = 0
    timer_train = time.time()
    while epoch < n_epochs:
        avg_costs = []
        timer = time.time()
        fish = ProgressFish(total=len(train_set_iterator))
        for i, (x, y) in enumerate(train_set_iterator, 1):
            fish.animate(amount=i)

            avg_cost = train_fn(x, y)
            if type(avg_cost) == list:
              avg_costs.append(avg_cost[0])
            else:
              avg_costs.append(avg_cost)

        mean_cost = numpy.mean(avg_costs)
        mean_train_error = numpy.mean(train_score())
        dev_error = numpy.mean(dev_score())
        print('epoch {} took {:.4f} seconds; '
              'avg costs: {:.4f}; train error: {:.4f}; '
              'dev error: {:.4f}'.format(epoch,time.time() - timer, mean_cost,
                                         mean_train_error, dev_error))

        if dev_error < best_dev_error:
            best_dev_error = dev_error
            best_params = [numpy.copy(p.get_value()) for p in params]
        epoch += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)

  def predict(self, x_test, y_test):
    test_set_iterator = DatasetMiniBatchIterator(self.rng, x_test, y_test, batch_size=batch_size, randomize=False)
    test_score = self._batch_score(test_set_iterator)

    print "Testing..."
    test_errors = numpy.mean(test_score())
    print "test error: {:.4f}".format(test_errors)


if __name__ == '__main__':
  nrows, ncols = 3813, 3
  batch_size = 50
  # x = numpy.arange(nrows * ncols).reshape((nrows, ncols))
  x = numpy.arange(nrows)
  print x[-1]
  rng = numpy.random.RandomState(123)
  x_iter = MiniBatchIteratorConstantBatchSize(rng, [x], batch_size=batch_size, randomize=False)

  for _ in xrange(100):
    for (batch,) in x_iter:
      assert len(batch) == batch_size

  print x_iter.n_samples