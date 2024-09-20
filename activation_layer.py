from Layers import Layer
from Param import LearnedParameter
from Activation import Activation
import numpy as np 
import config

# Activation Layer
class ActivationLayer(Layer):

  # initialize
  def __init__(self, in_features, out_features, activation=None, previous_activation_deriv=None):
    self.in_features = in_features
    self.out_features = out_features
    self.activation = None if activation == None else Activation(activation)
    self.previous_activation_deriv = previous_activation_deriv
    self.inputs = []
    self.optimizer = None

    # we randomly assign small values for the weights as the initiallization
    self.W = LearnedParameter(np.random.uniform(
      low=-np.sqrt(6. / (in_features + out_features)),
      high=np.sqrt(6. / (in_features + out_features)),
      size=(in_features, out_features)
    ))

    # normalize the weights to the mean and std values
    self.W.normalize(0, 0.1)

    # we set the size of bias as the size of output dimension
    self.b = LearnedParameter(np.zeros(out_features,))

  # call
  def __call__(self, x):
    return self.forward(x)

  # forward
  def forward(self, input, training=False):
    if training:
      self.inputs = input
    linear = np.atleast_2d(input).dot(np.atleast_2d(self.W.value)) + self.b.value
    output = linear if self.activation == None else self.activation(linear)

    return output

  # backward
  def backward(self, delta, batch_size):
    inputs = np.atleast_2d(self.inputs)
    self.W.grad = inputs.T.dot(np.atleast_2d(delta).reshape(inputs.shape[0], -1))
    self.b.grad = np.sum(delta, axis=0) if batch_size > 1 else delta

    # previous layer delta calculation
    delta = np.atleast_2d(delta).dot(np.atleast_2d(self.W.value).T)
    if self.previous_activation_deriv:
      delta = delta * self.previous_activation_deriv(self.inputs)

    return delta

  # zero grad
  def zero_grad(self):
    self.W.zero_grad()
    self.b.zero_grad()
    self.inputs = []

  # compile
  def compile(self, optimizer=None, learning_rate=config.LEARNING_RATE, weight_decay=None):
      self.optimizer = optimizer
      self.W.compile(optimizer, learning_rate, weight_decay)
      self.b.compile(optimizer, learning_rate, weight_decay)

  # update
  def update(self):
    # self.W.update()
    # self.b.update()
    self.W.value = self.W.value - 0.01 * self.W.grad
    self.b.value = self.b.value - 0.01 * self.b.grad.reshape(-1)


  # summary
  def summary(self):
    print("-----------------------------")
    print("Activation Layer:", self.activation)
    print("Number of Parameters:{} X {} + {} = {}".format(self.in_features, self.out_features, self.out_features, self.get_num_params()))

  # get number of parameters
  def get_num_params(self):
    return self.W.value.shape[0] * self.W.value.shape[1] + self.b.value.shape[0]

  # str
  def __str__(self):
    return "Layer: activation={}, in={}, out={}".format(self.activation, self.in_features, self.out_features)