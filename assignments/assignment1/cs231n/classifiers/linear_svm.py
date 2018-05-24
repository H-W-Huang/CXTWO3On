import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)  ## 对单个样本计算分值
    correct_class_score = scores[y[i]]   ## 记录下正确类别的分值
    ## 对于其他类别的分值
    ## 计算margin
    ## 若margin小于0,不记入loss
    diff_count = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        diff_count+= 1 
        dW[:,j] += X[i]  ### 注意此处用的是+=
    dW[:,y[i]] -= diff_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  


  print(dW.shape)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  ### 计算损失loss
  delta = 1.0
  num_train = X.shape[0]
  num_classes = W.shape[1]

  ## 首先计算得分
  scores = X.dot(W)
  print(">>> scores.shape:"+str(scores.shape))
  ## 计算损失
  ## 首先需要知道各个样本真实归类的分数，需要借助y 
  correct_class_score = scores[np.arange(num_train),y]

  print(">>> correct_class_score.shape :" + str(correct_class_score.shape))
  print(">>> correct_class_score.reshape(-1,1).shape :" + str(correct_class_score.reshape(-1,1).shape))
   ### 注意 max 和 maxium的区别
  ### max() 从一个array中找出最大值
  ### maximum(a,b) 则比较a和b两个元素的大小，返回较大的一个 
  margins = np.maximum(0, scores - correct_class_score.reshape(-1,1) + delta )
  print(">>> margins.shape:"+str(margins.shape))


  ## 此时还需要将margins中每一个样本的正确类别下的margin修改为0
  ## 其实也就是为了达到 j = y[i] 时，loss不做任何操作的效果
  margins[np.arange(num_train),y] = 0
  

  loss = np.sum(margins)
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ## 首先确定dW的shape,即W的shape
  dW = np.zeros(W.shape)
  print(">>> dW.shape:"+str(dW .shape))
  ### dW的计算需要margins的参与
  ## 计算出一个0-1构成的shape如W的矩阵
  # mask = margins > 0 ## True 和 False 构成
  # dW += 

  X_mask = np.zeros(margins.shape)
  print(">>> X_mask.shape:"+str(X_mask.shape))
  X_mask[margins > 0] = 1

  ## 对于每一个样本，计算 margin > 0 的非正确类别数目
  incorrect_count = np.sum(X_mask,axis =1)
  print(">>> incorrect_count.shape:"+str(incorrect_count.shape))
  ## 这一部分计算时参与到 j = y[i]，也就是说计算梯度时，处理预测分类和正确分类相同的情况
  ## 所以，
  X_mask[np.arange(num_train),y] = -incorrect_count
  print(">>> X_mask[0] == "+str(X_mask[0]))
  dW = X.T.dot(X_mask)
  print(">>> dW.shape:"+str(dW .shape))

  dW = dW/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


# Naive loss: 8.996912e+00 computed in 0.060159s
# scores.shape:(500, 10)
# correct_class_score.shape :(500,)
# correct_class_score.reshape(-1,1).shape :(500, 1)
# margins.shape:(500, 10)
# dW.shape:(3073, 10)
# X_mask.shape:(500, 10)
# incorrect_count.shape:(500,)
# dW.shape:(3073, 10)
# Vectorized loss: 8.996912e+00 computed in 0.004011s
# difference: -0.000000