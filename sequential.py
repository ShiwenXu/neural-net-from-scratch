from activation_layer import ActivationLayer
from batch_norm_layer import BatchNormLayer
from dropout_layer import DropoutLayer
import config 

# Sequential
class Sequential:

  # initialize
  def __init__(self, layers_in_out, activations, dropout=None, batch_normalization=False):
    self.layers = []

    if(len(layers_in_out) != len(activations) + 1):
      raise ValueError('Number of Layers In/Out={} should be one more than number of Activations={}'.format(len(layers_in_out), len(activations)))

    self.layers_in_out = layers_in_out
    self.activations = activations
    self.dropout = dropout
    self.batch_normalization = batch_normalization

    previous_activation_deriv = None

    # add input batch normalization
    if batch_normalization:
      input_bn = BatchNormLayer(layers_in_out[0])
      self.layers.append(input_bn)

    # define the layers
    for i in range(len(activations)):
      fc = ActivationLayer(in_features=layers_in_out[i], out_features=layers_in_out[i+1],
        activation=activations[i], previous_activation_deriv=previous_activation_deriv)

      previous_activation_deriv = fc.activation.f_deriv if fc.activation else None
      self.layers.append(fc)

      # check for layers before the ouput layer
      if i < len(activations) - 1:

        # add hidden layers batch normalization
        if batch_normalization:
          bn = BatchNormLayer(layers_in_out[ i + 1 ], config.ACTIVATION)
          previous_activation_deriv = bn.activation.f_deriv if bn.activation else None
          self.layers.append(bn)

        # add hidden layers dropout
        if dropout:
          d = DropoutLayer(dropout)
          self.layers.append(d)

    # call
  def __call__(self, x):
    return self.predict(x)

  # predict
  def predict(self, x):
    return self.forward(x)

  # forward
  def forward(self, input, training=False):
      for layer in self.layers:
        output = layer.forward(input, training)
        input = output
      return output

  # backward
  def backward(self, delta, batch_size):
    for layer in reversed(self.layers):
      delta = layer.backward(delta, batch_size)

  # Compile
  def compile(self, optimizer=None, learning_rate=config.LEARNING_RATE, weight_decay=None):
    for layer in self.layers:
      layer.compile(optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

  # zero grad
  def zero_grad(self):
    for layer in self.layers:
      layer.zero_grad()

  # update
  def update(self):
    for layer in self.layers:
      layer.update()

  # summary
  def summary(self):
    total_params = 0

    for layer in self.layers:
      layer.summary()
      total_params += layer.get_num_params()

    print("-----------------------------")
    print("Total Number of Layers:", len(self.layers))
    print("Total Number of Parameters:", total_params)

  # str
  def __str__(self):
    return "sequential: layers in/out={}, activations={}, dropout=, batch_normalization=".format(self.layers_in_out, self.activations, self.dropout, self.batch_normalization)