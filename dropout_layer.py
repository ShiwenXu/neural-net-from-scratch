from layer import Layer 
import numpy as np 
import config

# Dropout Layer
class DropoutLayer(Layer):

  # initialize
  def __init__(self, dropout):
    self.dropout = dropout
    self.mask = []

  # forward
  def forward(self, input, training=False):
    if training:
      self.mask = np.random.binomial(n=1, p=(1 - self.dropout), size=np.atleast_2d(input).shape[1]).reshape(1, -1)
      masked_input = input * self.mask
      return masked_input

    return input

  # backward
  def backward(self, delta, batch_size):
    return delta * self.mask

  # zero grad
  def zero_grad(self):
    self.mask = []

  # compile
  def compile(self, optimizer=None, learning_rate=config.LEARNING_RATE, weight_decay=None):
    pass

  # summary
  def summary(self):
    print("-----------------------------")
    print("Dropout Layer:", self.dropout)