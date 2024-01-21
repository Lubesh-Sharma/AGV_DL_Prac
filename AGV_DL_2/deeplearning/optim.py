import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""
def update(w, dw, config=None):
    """... (docstring remains the same)"""

    if config is None:  # Set default parameters if not provided
        config = {}
        config.setdefault('learning_rate', 1e-3)
        config.setdefault('momentum', 0.9)
        config.setdefault('decay_rate', 0.99)
        config.setdefault('eps', 1e-8)

    next_w = None

    # Choose the update rule to use
    update_rule = config.get('update_rule', 'sgd')
    if update_rule == 'sgd':
        next_w = w - config['learning_rate'] * dw
    elif update_rule == 'momentum':
        v = config.get('velocity', np.zeros_like(w))  # Initialize velocity if needed
        v = config['momentum'] * v - config['learning_rate'] * dw
        next_w = w + v
    elif update_rule == 'rmsprop':
        cache = config.get('cache', np.zeros_like(w))  # Initialize cache if needed
        cache = config['decay_rate'] * cache + (1 - config['decay_rate']) * dw**2
        next_w = w - config['learning_rate'] * dw / (np.sqrt(cache) + config['eps'])
    else:
        raise ValueError(f"Unknown update rule: {update_rule}")

    return next_w, config

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    #############################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw  # Update velocity
    next_w = w + v  # Apply velocity to weights                                 #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    #############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x   #
    # in the next_x variable. Don't forget to update cache value stored in      #
    # config['cache'] and to use the epsilon scalar to avoid dividing by zero.  #
    #############################################################################
    cache = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx**2  # Update cache
    next_x = x - config['learning_rate'] * dx / (np.sqrt(cache) + config['epsilon'])  # Update weights
    config['cache'] = cache 
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config and to use the epsilon scalar to avoid dividing by zero. #
    # Adam update formula
    #############################################################################
    t = config['t'] + 1  # Increment iteration number
    m = config['beta1'] * config['m'] + (1 - config['beta1']) * dx  # Update biased first moment estimate
    mt = m / (1 - config['beta1']**t)  # Compute bias-corrected first moment estimate
    v = config['beta2'] * config['v'] + (1 - config['beta2']) * dx**2  # Update biased second moment estimate
    vt = v / (1 - config['beta2']**t)  # Compute bias-corrected second moment estimate
    next_x = x - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])  # Update weights
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
	config['t'] = t  # Store updated iteration number for the next iteration
    config['m'] = m  # Store updated biased first moment estimate
    config['v'] = v  # Store updated biased second moment estimate
    return next_x, config
