import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    tmp_exp = 0
    for j in range(num_class):
      tmp_exp += np.exp(X[i].dot(W[:,j]))
    loss += -1*np.log(np.exp(X[i].dot(W[:,y[i]]))/tmp_exp)
    for j in range(num_class):
      if j == y[i]:
        dW[:,j] += (np.exp(X[i].dot(W[:,j]))/tmp_exp-1)*X[i];
      else:
        dW[:,j] += np.exp(X[i].dot(W[:,j]))/tmp_exp*X[i];
    pass

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW /= num_train
  dW += reg*dW

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  exp_loss = np.exp(X.dot(W))/np.sum(np.exp(X.dot(W)),axis=1).reshape((num_train,1))
  loss = np.sum(-1*np.log(exp_loss[np.arange(num_train),y].reshape(num_train,1)))
  loss /= num_train
  #loss = -1*np.log(loss)
  loss += 0.5 * reg * np.sum(W ** 2)
  #dW_correct = np.zeros(dW.shape)
  exp_loss[np.arange(num_train),y] -= 1
  dW = X.T.dot(exp_loss)
  dW /= num_train
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

