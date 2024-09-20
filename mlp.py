import numpy as np 
import config 
import math
from loss_func import LossFunction

# Main MLP Class
class MLP:

    # initialize
    def __init__(self, sequential):
      self.sequential = sequential

    # summary
    def summary(self):
      print("=============================")
      print("MLP Summary:")
      self.sequential.summary()
      print("=============================")

    # Compile
    def compile(self, optimizer=None, learning_rate=config.LEARNING_RATE, weight_decay=None):
      self.sequential.compile(optimizer, learning_rate, weight_decay)

    # call
    def __call__(self, x):
      return self.predict(x)

    # predict
    def predict(self, x):
      return self.sequential(x)

    # accuracy
    def accuracy(self, preds, target):
      target = target.reshape(-1)
      predict = []
      for i in preds:
        predict.append(np.argmax(i))
      predict = np.array(predict)

      return np.mean(predict == target)

    # shuffle examples
    def shuffle_examples(self, x, y):
      x = np.array(x)
      y = np.array(y)
      rand_indexes = np.random.permutation(x.shape[0])
      shuffled_x = x[rand_indexes]
      shuffled_y = y[rand_indexes]

      return shuffled_x, shuffled_y

    # fit
    def fit(self, train_x, train_y, epochs, loss, batch_size=1, validation_data=None, early_stop=False):
      n_batches = math.ceil(len(train_x)/batch_size)
      loss_func = LossFunction(loss)
      print("Start fitting with epoch={}, batches={}, loss={}...".format(epochs, n_batches, loss))

      last_val_loss = None
      count = 0
      training_loss = []

      if validation_data and early_stop:
        validate_x = validation_data[0]
        validate_y = validation_data[1]
        preds_y = self.sequential(validate_x)
        delta, val_loss = loss_func(preds_y, validate_y)
        last_val_loss = val_loss

      for epoch in range(epochs):
        print("Start Epoch {}...".format(epoch))
        total_loss = []
        train_acc = []
        # shuffle the data to avoid overfitting
        shuffled_x, shuffled_y = self.shuffle_examples(train_x, train_y)

        for batch in range(n_batches):
          batch_start = batch * batch_size
          batch_end = min(batch_start + batch_size, len(train_x))
          batch_size = batch_end - batch_start
          batch_x = shuffled_x[batch_start:batch_end]
          batch_y = shuffled_y[batch_start:batch_end]

          # mini-batch: calculate gradients & updates the weights after each batch
          preds = self.sequential.forward(batch_x, training=True)
          # calculate accuracy
          batch_acc = self.accuracy(preds, batch_y)
          # calculate batch error, loss and delta
          error, batch_loss = loss_func(preds, batch_y)
          total_loss.append(batch_loss)
          train_acc.append(batch_acc)
          delta = -np.atleast_2d(error).reshape(batch_size, -1)
          output_layer_activation = self.sequential.layers[-1].activation
          output_layer_derivative = output_layer_activation.f_deriv if output_layer_activation else None
          delta = delta * np.array(output_layer_derivative(preds)).reshape(delta.shape) if output_layer_derivative else delta

          # compute the gradients of the average loss with respect to the weights
          self.sequential.backward(delta, batch_size)
          self.sequential.update()
          self.sequential.zero_grad()

        epoch_loss = np.mean(total_loss)
        epoch_acc = np.mean(train_acc)
        print("epoch loss: {:.4f}".format(epoch_loss))
        print('train_acc: {:.4f}'.format(epoch_acc))
        training_loss.append(epoch_loss.tolist())
        print("-----------------------------")

        if(validation_data):
          validate_x = validation_data[0]
          validate_y = validation_data[1]
          y_hat = self.sequential(validate_x)
          delta, val_loss = loss_func(y_hat, validate_y)
          print("validation loss: {:.4f}".format(val_loss))

          if early_stop and epoch % config.EPOCH_EVAL == 0:
            if val_loss > last_val_loss:
              count += 1
            else:
              last_val_loss = val_loss
              count = 0

          if count == config.MAX_COUNT:
            print('early stopped!!!')
            break

      print("End fitting")
      print("=============================")

      return training_loss