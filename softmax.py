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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train,dim=X.shape
  num_class=W.shape[1]
  scores=np.zeros((num_train,num_class))

  for i in range(num_train):
    scores[i]=X[i].dot(W)
    scores[i] -=np.max(scores[i,:])   #提高计算中的数值稳定性
    sum=0.0
    for j in range(num_class):
      sum +=np.exp(scores[i,j])
    #sum=np.sum(np.exp(score[i]))
    loss += np.log(sum)-scores[i,y[i]]
    for j in range(num_class):
      if j==y[i]:
        dW[:,j] +=(np.exp(scores[i,j])/sum-1)*X[i]  #这里loss function里有的是以e为底的对数
      else:
        dW[:,j] +=(np.exp(scores[i,j])/sum)*X[i]


  loss /=num_train
  dW /=num_train
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train,dim=X.shape
  num_class=W.shape[1]
  scores=X.dot(W)
  scores -=np.max(scores,axis=1).reshape(num_train,-1)

  '''
  for i in range(num_train):
    sum=np.sum(np.exp(scores[i]))
    #loss +=-np.log10(np.exp(scores[i,y[i]])/sum)
    loss +=np.log(sum)-scores[i,y[i]]
    for j in range(num_class):
      if j==y[i]:
        dW[:,j] +=(np.exp(scores[i,j])/(sum)-1)*X[i]
      else:
        dW[:,j] +=(np.exp(scores[i,j])/(sum))*X[i]
  '''

  scores_label=scores[range(num_train),y]   #fancy索引，得到的是one-dimension的array
  scores_sum=np.sum(np.exp(scores),axis=1)
  loss +=np.sum(np.log(scores_sum)-scores_label)

  for i in range(num_train):
    #dW +=np.exp(scores)/scores_sum.reshape(num_train,-1)*X[i].reshape(num_train,-1)
    dW += X[i].reshape(dim, -1).dot((np.exp(scores[i]) / (scores_sum[i])).reshape(-1,num_class))
    dW[:,y[i]] -=X[i]


  loss /=num_train
  dW /=num_train
  loss +=reg*np.sum(W*W)
  dW +=2*reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

