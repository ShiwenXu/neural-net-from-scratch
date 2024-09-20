import config
import numpy as np 


# Optimizer
class Optimizer:

  # SGD
  def __sgd(self, grad):
    return self.lr * grad

  # Momentum SGD
  def __momentum_sgd(self, grad):
    self.s = config.MOMENTUM * self.s + np.multiply(self.lr, grad)
    return self.s

  # Adam
  def __adam(self, grad):
    self.t += 1
    self.s = np.multiply(config.ADAM_RHO1 , self.s ) + np.multiply(( 1 - config.ADAM_RHO1) , grad)
    self.r = np.multiply(config.ADAM_RHO2 , self.r ) + np.multiply(( 1 - config.ADAM_RHO2) , np.multiply(grad, grad))
    st = self.s / ( 1 - config.ADAM_RHO1**self.t)
    rt = self.r / ( 1 - config.ADAM_RHO2**self.t)
    return self.lr * st / (np.sqrt( rt + EPS))

  # initialize
  def __init__(self, optimizer=config.OPTIMIZER, learning_rate=config.LEARNING_RATE, param_shape=None, weight_decay=None):
    self.name = optimizer
    self.lr = learning_rate
    self.weight_decay = weight_decay

    if optimizer == 'sgd':
        self.optimizer_f = self.__sgd
    elif optimizer == 'momentum_sgd':
        self.optimizer_f = self.__momentum_sgd
        self.s = np.zeros(param_shape)
    elif optimizer == 'adam':
        self.optimizer_f = self.__adam
        self.t = 0
        self.s = np.zeros(param_shape)
        self.r = np.zeros(param_shape)
    else:
        raise ValueError('Unknown Optimizer')

  # Call
  def __call__(self, grad, param_value=None):
    optimized_grad = self.optimizer_f(grad)

    if self.weight_decay:
      optimized_grad = optimized_grad + np.multiply(self.lr * self.weight_decay, param_value)

    return optimized_grad

  # str
  def __str__(self):
    return self.name