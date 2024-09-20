from abc import ABC, abstractmethod

# Layer
class Layer(ABC):

  # call
  def __call__(self, x):
    return self.forward(x)

  # forward
  @abstractmethod
  def forward(self, input, training=False):
    pass

  # backward
  @abstractmethod
  def backward(self, delta, batch_size):
    return delta

  # summary
  @abstractmethod
  def summary(self):
    pass

  # compile
  @abstractmethod
  def compile(self, optimizer):
    pass

  # zero grad
  @abstractmethod
  def zero_grad(self):
    pass

  # update
  def update(self):
    pass

  # get number of parameters
  def get_num_params(self):
    return 0