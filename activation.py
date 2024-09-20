import numpy as np 
import config 


# Activation
class Activation(object):

  # tanh
  def __tanh(self, x):
      return np.tanh(x)

  # tanh deriv
  def __tanh_deriv(self, a):
      return 1.0 - a**2

  # logistic
  def __logistic(self, x):
      return 1.0 / (1.0 + np.exp(-x))

  # logistic deriv
  def __logistic_deriv(self, a):
      return  a * (1 - a )

  # relu
  def __relu(self, x):
    return np.maximum(0, x)

  # relu deriv
  def __relu_deriv(self, a):
    return np.where(a>0, 1, 0)

  #TODO find a reference and change this code received from chatgpt
  #TODO to be tested with mlp
  # reference: https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py
  # gelu
  def __gelu(self, x):
    c = np.sqrt(2/np.pi)
    return 0.5 * x * (1 + np.tanh(c*(x+0.044715*np.power(x, 3))))

  # gelu deriv
  def __gelu_deriv(self, a):
    c = np.sqrt(2/np.pi)
    return 0.5 * (1 + np.tanh(c*(a+0.044715*np.power(a, 3)))) + 0.5 * a * c * (1 - np.power(np.tanh(c*(a+0.044715*np.power(a, 3))), 2)) * (1 + 0.134145 * np.power(a, 2) + 0.044715 * np.power(a, 4))

  # sigmoid
  def __sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  # sigmoid deriv: sigmoid(input)*(1-sigmoid(input)) where sigmoid(input) is already calculated in the forward pass
  def __sigmoid_deriv(self, activated_output):
    return activated_output * (1 - activated_output)

  # softmax
  def __softmax(self, x):
    numerator = np.exp(x - np.max(x))
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

  # softmax deriv
  # this code is based on https://stackoverflow.com/questions/54976533/derivative-of-softmax-function-in-python
  def __softmax_deriv(self, activated_output):
  #   jacobian_m = np.diag(activated_output)

  #   for i in range(len(jacobian_m)):
  #     for j in range(len(jacobian_m)):
  #       if i == j:
  #         jacobian_m[i][j] = activated_output[i] * (1-activated_output[i])
  #       else:
  #         jacobian_m[i][j] = -activated_output[i]*activated_output[j]

  #   return  jacobian_m

    # softmax deriv has been combined with cross entropy loss deriv
    return np.ones(np.array(activated_output).shape)

  # initialize
  def __init__(self, activation=config.ACTIVATION):
      self.name = activation
      if activation == 'relu':
          self.f = self.__relu
          self.f_deriv = self.__relu_deriv
      elif activation == 'softmax':
          self.f = self.__softmax
          self.f_deriv = self.__softmax_deriv
      elif activation == 'sigmoid':
          self.f = self.__sigmoid
          self.f_deriv = self.__sigmoid_deriv
      elif activation == 'tanh':
          self.f = self.__tanh
          self.f_deriv = self.__tanh_deriv
      elif activation == 'logistic':
          self.f = self.__logistic
          self.f_deriv = self.__logistic_deriv
      elif activation == 'gelu':
          self.f = self.__gelu
          self.f_deriv = self.__gelu_deriv
      else:
          raise ValueError('Unknown Activation')

  # call
  def __call__(self, x):
    return self.f(x)

  # str
  def __str__(self):
    return self.name