import numpy as np 
import config

# Loss Function
class LossFunction(object):

  # cross-entropy
  def __cross_entropy(self, preds, target):
    # Create the one-hot encoding
    size = np.array(target).shape[0]
    preds = np.array(preds).reshape(size, -1)
    y_true_one_hot =  np.eye(preds.shape[1])[target].reshape(size, -1)
    # Calculate cross entropy loss
    loss = -np.sum(y_true_one_hot * np.log(preds+config.EPS)) / size
    error = y_true_one_hot - preds

    return error, loss

  # MSE
  def __mse(self, preds, target):
    target = np.array(target)
    preds = np.array(preds).reshape(target.shape)
    error =  target - preds
    loss = error ** 2

    return error, loss

  # initialize
  def __init__(self, loss=config.LOSS_FUNC):
      self.name = loss
      if loss == 'cross_entropy':
        self.loss_f = self.__cross_entropy
      elif loss == 'mse':
        self.loss_f = self.__mse
      else:
        raise ValueError('Unknown Loss')

  # call
  def __call__(self, preds, target):
      return self.loss_f(preds, target)

  # str
  def __str__(self):
    return self.name