
import numpy as np 
from Optim import Optimizer
import config

# Learned Parameter
class LearnedParameter:

  # initialize
  def __init__(self, value=None):
    self.value = value
    self.grad = np.zeros(self.value.shape)
    self.optimizer = None

  # zero grad
  def zero_grad(self):
    self.grad = np.zeros(self.value.shape)

  # compile
  def compile(self, optimizer, learning_rate=config.LEARNING_RATE, weight_decay=None):
    self.optimizer = Optimizer(
        config.OPTIMIZER if optimizer == None else optimizer,
        learning_rate=learning_rate,
        param_shape=self.value.shape,
        weight_decay=weight_decay)

  # update
  def update(self):
    self.value -= self.optimizer(self.grad, param_value=self.value).reshape(self.value.shape)

  # normalize
  def normalize(self, mean, std):
    old_mean = np.mean(self.value)
    old_std = np.std(self.value)
    self.value = np.interp(self.value, (old_mean - old_std, old_mean + old_std), (mean - std, mean + std))

  # str
  def __str__(self):
    return "parameter: {}".format(self.value.shape)