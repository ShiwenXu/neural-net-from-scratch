from parameter import LearnedParameter
import numpy as np 
from layer import Layer 
from activation import Activation
import config

# Batch Normalization Layer
class BatchNormLayer(Layer):

  # initialize
  def __init__(self, num_features, activation=None, previous_activation_deriv=None):
    self.num_features = num_features
    self.activation = None if activation == None else Activation(activation)
    self.previous_activation_deriv = previous_activation_deriv
    self.gamma = LearnedParameter(np.ones(num_features))
    self.beta = LearnedParameter(np.zeros(num_features))
    self.mean = np.zeros(num_features)
    self.var = np.ones(num_features)
    self.x_norm = None
    self.x_centered = None
    self.batch_std = None
    self.inputs = None
    self.optimizer = None

  # forward
  def forward(self, input, training=False):
    x = np.array(input)
    y = 0

    # training mode
    if training:
      batch_mean = x.mean(axis=0)
      batch_var = x.var(axis=0)

      # momentum mean and var has been used instead for ease as they don't require to keep track of previous batches
      self.mean = config.MOMENTUM * self.mean + ( 1 - config.MOMENTUM ) * batch_mean
      self.var = config.MOMENTUM * self.var + ( 1 - config.MOMENTUM ) * batch_var

      batch_std = np.sqrt(batch_var + config.EPS)
      x_centered = x - batch_mean
      x_norm = x_centered / batch_std
      self.x_norm = x_norm
      self.x_centered = x_centered
      self.batch_std = batch_std
      self.inputs = x
      y = np.atleast_2d(x_norm).dot(np.atleast_2d(self.gamma.value).T) + self.beta.value

    # prediction mode
    else:
      x_norm = (x - self.mean) / np.sqrt(self.var + config.EPS)
      y = np.atleast_2d(x_norm).dot(np.atleast_2d(self.gamma.value).T) + self.beta.value

    output = y if self.activation == None else self.activation(y)

    return output


  # backward
  def backward(self, delta, batch_size):
    x_norm = np.atleast_2d(self.x_norm)
    self.gamma.grad = x_norm.T.dot(np.atleast_2d(delta).reshape(x_norm.shape[0], -1)).sum(axis=0)
    self.beta.grad = np.sum(delta, axis=0) if batch_size > 1 else delta

    x_norm_grad = delta * self.gamma.value
    x_centered_grad = x_norm_grad / self.batch_std
    mean_grad = -(x_centered_grad.sum(axis=0) + 2 / batch_size * self.x_centered.sum(axis=0))
    std_grad = (x_norm_grad * self.x_centered * -self.batch_std**(-2)).sum(axis=0)
    var_grad = std_grad / 2 / self.batch_std
    delta = x_centered_grad + (mean_grad + var_grad * 2 * self.x_centered) / batch_size

    if self.previous_activation_deriv:
      delta = x_norm_grad * self.previous_activation_deriv(self.inputs)

    return delta

  # zero grad
  def zero_grad(self):
    self.gamma.zero_grad()
    self.beta.zero_grad()
    self.x_norm = None
    self.x_centered = None
    self.batch_std = None
    self.inputs = None

  # compile
  def compile(self, optimizer=None, learning_rate=config.LEARNING_RATE, weight_decay=None):
      self.optimizer = optimizer
      self.gamma.compile(optimizer, learning_rate, weight_decay)
      self.beta.compile(optimizer, learning_rate, weight_decay)

  # update
  def update(self):
    self.gamma.update()
    self.beta.update()

  # get number of parameters
  def get_num_params(self):
    return self.num_features * 2

  # summary
  def summary(self):
    print("-----------------------------")
    print("Number of Parameters:{} + {} = {}".format(self.num_features, self.num_features, self.get_num_params()))
    print("Batch Normalization Layer: number of features={}, activation={}".format(self.num_features, self.activation))