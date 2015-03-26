#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

from deepy.trainers.trainer import SGDTrainer, NeuralTrainer
import theano
import numpy as np
import theano.tensor as T


class NAG(SGDTrainer):
    '''Optimize using Nesterov's Accelerated Gradient (NAG).
    The basic difference between NAG and "classical" momentum in SGD
    optimization approaches is that NAG computes the gradients at the position
    in parameter space where "classical" momentum would put us at the *next*
    step. In symbols, the classical method with momentum m and learning rate a
    updates parameter p by blending the current "velocity" with the current
    gradient:
        v_t+1 = m * v_t - a * grad(p_t)
        p_t+1 = p_t + v_t+1
    while NAG adjusts the update by blending the current "velocity" with the
    next-step gradient (i.e., the gradient at the point where the velocity
    would have taken us):
        v_t+1 = m * v_t - a * grad(p_t + m * v_t)
        p_t+1 = p_t + v_t+1
    The difference here is that the gradient is computed at the place in
    parameter space where we would have stepped using the classical
    technique, in the absence of a new gradient.
    In theory, this helps correct for oversteps during learning: If momentum
    would lead us to overshoot, then the gradient at that overshot place
    will point backwards, toward where we came from. (For details see
    Sutskever, Martens, Dahl, and Hinton, ICML 2013, "On the importance of
    initialization and momentum in deep learning.")
    '''

    def __init__(self, network, **kwargs):
        # due to the way that theano handles updates, we cannot update a
        # parameter twice during the same function call. so, instead of handling
        # everything in the updates for self.f_learn(...), we split the
        # parameter updates into two function calls. the first "prepares" the
        # parameters for the gradient computation by moving the entire model one
        # step according to the current velocity. then the second computes the
        # gradient at that new model position and performs the usual velocity
        # and parameter updates.

        self.params = network.params(**kwargs)
        self.momentum = kwargs.get('momentum', 0.5)

        # set up space for temporary variables used during learning.
        self._steps = []
        self._velocities = []
        for param in self.params:
            v = param.get_value()
            n = param.name
            self._steps.append(theano.shared(np.zeros_like(v), name=n + '_step'))
            self._velocities.append(theano.shared(np.zeros_like(v), name=n + '_vel'))

        # step 1. move to the position in parameter space where we want to
        # compute our gradient.
        prepare = []
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            prepare.append((step, self.momentum * velocity))
            prepare.append((param, param + step))

        logging.info('compiling NAG adjustment function')
        self.f_prepare = theano.function([], [], updates=prepare)

        super(NAG, self).__init__(network, **kwargs)

    def learning_updates(self):
        # step 2. record the gradient here.
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            yield velocity, step - self.learning_rate * TT.grad(self.J, param)

        # step 3. update each of the parameters, removing the step that we took
        # to compute the gradient.
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            yield param, param + velocity - step

    def train_minibatch(self, *x):
        self.f_prepare()
        return self.learning_func(*x)


class Rprop(SGDTrainer):
    '''Trainer for neural nets using resilient backpropagation.
    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference is that in Rprop, only the signs of the partial derivatives are
    taken into account when making parameter updates. That is, the step size for
    each parameter is independent of the magnitude of the gradient for that
    parameter.
    To accomplish this, Rprop maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient of J with respect to that parameter
    over time. Whenever two consecutive gradients for a parameter have the same
    sign, the learning rate for that parameter increases, and whenever the signs
    disagree, the learning rate decreases. This has a similar effect to
    momentum-based SGD methods but effectively maintains parameter-specific
    momentum values.
    The implementation here actually uses the "iRprop-" variant of Rprop
    described in Algorithm 4 from Igel and Huesken (2000), "Improving the Rprop
    Learning Algorithm." This variant resets the running gradient estimates to
    zero in cases where the previous and current gradients have switched signs.
    '''

    def __init__(self, network, **kwargs):
        self.step_increase = kwargs.get('rprop_increase', 1.01)
        self.step_decrease = kwargs.get('rprop_decrease', 0.99)
        self.min_step = kwargs.get('rprop_min_step', 0.)
        self.max_step = kwargs.get('rprop_max_step', 100.)
        super(Rprop, self).__init__(network, **kwargs)

    def learning_updates(self):
        step = self.learning_rate
        self.grads = []
        self.steps = []
        for param in self.params:
            v = param.get_value()
            n = param.name
            self.grads.append(theano.shared(np.zeros_like(v), name=n + '_grad'))
            self.steps.append(theano.shared(np.zeros_like(v) + step, name=n + '_step'))
        for param, step_tm1, grad_tm1 in zip(self.params, self.steps, self.grads):
            grad = TT.grad(self.J, param)
            test = grad * grad_tm1
            same = TT.gt(test, 0)
            diff = TT.lt(test, 0)
            step = TT.minimum(self.max_step, TT.maximum(self.min_step, step_tm1 * (
                TT.eq(test, 0) +
                same * self.step_increase +
                diff * self.step_decrease)))
            grad = grad - diff * grad
            yield param, param - TT.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step



class Scipy(NeuralTrainer):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, method, **kwargs):
        super(Scipy, self).__init__(network, **kwargs)

        self.method = method
        self.iterations = kwargs.get('num_updates', 100)

        logging.info('compiling gradient function')
        self.f_grad = theano.function(network.inputs, T.grad(self.J, self.params))

    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        return np.mean([self.evaluation_func(*x)[0] for x in train_set])

    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in train_set:
            for i, g in enumerate(self.f_grad(*x)):
                grads[i].append(np.asarray(g))
        return self.arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def train(self, train_set, valid_set=None, **kwargs):
        def display(x):
            self.set_params(self.flat_to_arrays(x))
            costs = np.mean([self.evaluation_func(*x) for x in train_set], axis=0)
            cost_desc = ' '.join(
                '%s=%.2f' % el for el in zip(self.cost_names, costs))
            logging.info('scipy.%s %i %s', self.method, i + 1, cost_desc)

        for i in range(self.iterations):
            try:
                if not self.evaluate(i, valid_set):
                    logging.info('patience elapsed, bailing out')
                    break
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            try:
                res = scipy.run.minimize(
                    fun=self.function_at,
                    jac=self.gradient_at,
                    x0=self.arrays_to_flat(self.best_params),
                    args=(train_set, ),
                    method=self.method,
                    callback=display,
                    options=dict(maxiter=self.validation_frequency),
                )
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            self.set_params(self.flat_to_arrays(res.x))

            yield {'J': res.fun}

        self.set_params(self.best_params)


class LM(NeuralTrainer):
    '''Levenberg-Marquardt trainer for neural networks.
    Based on the description of the algorithm in "Levenberg-Marquardt
    Optimization" by Sam Roweis.
    '''

    def __init__(self, network, **kwargs):
        raise NotImplementedError
#
#
# class HF(Trainer):
#     '''The hessian free trainer shells out to an external implementation.
#     hf.py was implemented by Nicholas Boulanger-Lewandowski and made available
#     to the public (yay !). If you don't have a copy of the module handy, this
#     class will attempt to download it from github.
#     '''
#
#     URL = 'https://raw.github.com/boulanni/theano-hf/master/hf.py'
#
#     def __init__(self, network, **kwargs):
#         import hf
#
#         self.params = network.params(**kwargs)
#         self.opt = hf.hf_optimizer(
#             self.params,
#             network.inputs,
#             network.y,
#             [network.J(**kwargs)] + [mon for _, mon in network.monitors],
#             network.hiddens[-1] if isinstance(network, recurrent.Network) else None)
#
#         # fix mapping from kwargs into a dict to send to the hf optimizer
#         kwargs['validation_frequency'] = kwargs.pop('validate', 1 << 60)
#         try:
#             func = self.opt.train.__func__.__code__
#         except: # Python 2.x
#             func = self.opt.train.im_func.func_code
#         for k in set(kwargs) - set(func.co_varnames[1:]):
#             kwargs.pop(k)
#         self.kwargs = kwargs
#
#     def train(self, train_set, valid_set=None, **kwargs):
#         self.set_params(self.opt.train(
#             train_set, kwargs['cg_set'], validation=valid_set, **self.kwargs))
#         yield {'J': -1}


class Sample(NeuralTrainer):
    '''This trainer replaces network weights with samples from the input.'''

    @staticmethod
    def reservoir(xs, n):
        '''Select a random sample of n items from xs.'''
        pool = []
        for i, x in enumerate(xs):
            if len(pool) < n:
                pool.append(x / np.linalg.norm(x))
                continue
            j = rng.randint(i + 1)
            if j < n:
                pool[j] = x / np.linalg.norm(x)
        # if the pool still has fewer than n items, pad with distorted random
        # duplicates from the source data.
        L = len(pool)
        S = np.std(pool, axis=0)
        while len(pool) < n:
            x = pool[rng.randint(L)]
            pool.append(x + S * rng.randn(*x.shape))
        return np.array(pool, dtype=pool[0].dtype)

    def __init__(self, network, **kwargs):
        self.network = network

    def train(self, train_set, valid_set=None, **kwargs):
        ifci = itertools.chain.from_iterable

        # set output (decoding) weights on the network.
        last = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        samples = ifci(last(t) for t in train_set)
        for w in self.network.weights:
            k, n = w.get_value(borrow=True).shape
            if w.name.startswith('W_out_'):
                arr = np.vstack(Sample.reservoir(samples, k))
                logging.info('setting weights for %s: %d x %d <- %s', w.name, k, n, arr.shape)
                w.set_value(arr / np.sqrt((arr * arr).sum(axis=1))[:, None])

        # set input (encoding) weights on the network.
        first = lambda x: x[0] if isinstance(x, (tuple, list)) else x
        samples = ifci(first(t) for t in train_set)
        for i, h in enumerate(self.network.hiddens):
            if i == len(self.network.weights):
                break
            w = self.network.weights[i]
            m, k = w.get_value(borrow=True).shape
            arr = np.vstack(Sample.reservoir(samples, k)).T
            logging.info('setting weights for %s: %d x %d <- %s', w.name, m, k, arr.shape)
            w.set_value(arr / np.sqrt((arr * arr).sum(axis=0)))
            samples = ifci(self.network.feed_forward(first(t))[i-1] for t in train_set)

        yield {'J': -1}


class Layerwise(NeuralTrainer):
    '''This trainer adapts parameters using a variant of layerwise pretraining.
    In this variant, we create "taps" at increasing depths into the original
    network weights, training only those weights that are below the tap. So, for
    a hypothetical binary classifier network with layers [3, 4, 5, 6, 2], we
    would first insert a tap after the first hidden layer (effectively a binary
    classifier in a [3, 4, 2] configuration) and train just that network. Then
    we insert a tap at the next layer (effectively training a [3, 4, 5, 2]
    classifier, re-using the trained weights for the 3x4 layer), and so forth.
    By inserting taps into the original network, we preserve all of the relevant
    settings of noise, dropouts, loss function and the like, in addition to
    removing the need for copying trained weights around between different
    Network instances.
    '''

    def __init__(self, network, factory, *args, **kwargs):
        self.network = network
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        '''Train a network using a layerwise strategy.
        Parameters
        ----------
        train_set : :class:`theanets.Dataset`
            A training set to use while training the weights in our network.
        valid_set : :class:`theanets.Dataset`
            A validation set to use while training the weights in our network.
        Returns
        -------
        Generates a series of cost values as the network weights are tuned.
        '''
        net = self.network

        y = net.y
        hiddens = list(net.hiddens)
        weights = list(net.weights)
        biases = list(net.biases)

        nout = len(biases[-1].get_value(borrow=True))
        nhids = [len(b.get_value(borrow=True)) for b in biases]
        for i in range(1, len(weights) + 1 if net.tied_weights else len(nhids)):
            net.hiddens = hiddens[:i]
            if net.tied_weights:
                net.weights = [weights[i-1]]
                net.biases = [biases[i-1]]
                for j in range(i - 1, -1, -1):
                    net.hiddens.append(T.dot(net.hiddens[-1], weights[j].T))
                net.y = net._output_func(net.hiddens.pop())
            else:
                W, b, _ = net.create_layer(nhids[i-1], nout, 'layerwise')
                net.weights = [weights[i-1], W]
                net.biases = [biases[i-1], b]
                net.y = net._output_func(T.dot(hiddens[i-1], W) + b)
            logging.info('layerwise: training weights %s', net.weights[0].name)
            trainer = self.factory(net, *self.args, **self.kwargs)
            for costs in trainer.train(train_set, valid_set):
                yield costs

        net.y = y
        net.hiddens = hiddens
        net.weights = weights
        net.biases = biases