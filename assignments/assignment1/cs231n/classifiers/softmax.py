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
  ## 数据集X中有多少个样本，就有多少次训练，就需要计算多少次损失
  num_train = X.shape[0]
  ## 参数W中有几列，就有多少种类别（一行对应一种类别的模式）
  # W: A numpy array of shape (D, C) containing weights.
  num_classes = W.shape[1]   ## ERROR log， W.shape[0] -> W.shape[1]
    
  print(num_train)
  for i in range(num_train):
    ### 第一步，计算fi，其中fi=WX
    fi = X[i].dot(W)  ## shape = (C)

    ## When you’re writing code for computing the Softmax function in practice, 
    # the intermediate terms efyi and ∑jefj may be very large due to the exponentials. 
    # Dividing large numbers can be numerically unstable, 
    # so it is important to use a normalization trick.
    # 数字太多可能会有溢出等问题，导致计算结果不稳定
    # 因此正则化 
    fi -= np.max(fi)

    ### 此时就可以根据损失函数的定义计算loss了
    ### 注意，loss函数可以有很多种
    ### 此处使用的是 交叉熵损失 而不是用 hinge loss
    # http://cs231n.github.io/linear-classify/
    loss_i = - fi[y[i]] + np.log(np.sum(np.exp(fi)))
    #print(loss_i)
    loss += loss_i

    ### 对当前样本，求dW
    ## 计算dW时，正确类别的部分需要减去 X[i]
    ## ！！！！！
    
    ### 根据softmax的定义编写
    #dW[:,y[i]] -= X[i]
    ### 通过分别求取 W 上的偏导后组合而成
    ### softmax参与到偏导的计算中
    for j in range(num_classes):
      softmax_output = np.exp(fi[j])/ np.sum(np.exp(fi))
      if j == y[i]:
        dW[:,j] -= X[i]
      dW[:,j] +=  softmax_output*X[i]
      
  print(loss)
  loss = loss / num_train +  0.5 * reg * np.sum(W * W)  ## 引入L2正则化
  dW = dW / num_train + reg * W


  #pass
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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W) # shape = (N,C)
  ## 一行表示一个样本，列元素分别为其在各个分类上的得分
  ## 正则化
  ## ERROR LOG
  ###  scores -= scores - np.max(scores,axis=1,keepdims=True) 
  ### ==> scores -=  np.max(scores,axis=1,keepdims=True) 
  scores -= np.max(scores,axis=1,keepdims=True)  ## shape = (N,C)
  print(scores.shape)
  softmax_output = np.exp(scores) / np.sum(np.exp(scores),axis = 1,keepdims=True)
  print(softmax_output.shape)
  ## 通过 softmax 可以计算 loss 结果
  loss = -np.sum(np.log(softmax_output[range(num_train),list(y)]))
  ## softmax_output[range(num_train),list(y)]
  ## 意思是，
  ## 对softmax，从0...i...num_train
  ## 索引 softmax_output[i,y[i]]
  ## 此处y的长度和num_train相等
  loss = loss / num_train + 0.5 * reg * np.sum(W*W)

  ## 计算dW，还是和一步步计算的思路一样，正确类别的部分需要减去 X[i]
  ## 构造一个从Softmax结果修改出来的矩阵
  dS = softmax_output.copy()
  ## 对每一个样本，对其正确分类部分的softmax结果-1
  dS[range(num_train),list(y)] += -1
  # 单步的计算如下
  # for j in range(num_classes):
  #   softmax_output = np.exp(fi[j])/ np.sum(np.exp(fi))
  #   if j == y[i]:
  #     dW[:,j] -= X[i]
  #   dW[:,j] +=  softmax_output*X[i]
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg*W


  # ans
  # num_classes = W.shape[1]
  # num_train = X.shape[0]
  # scores = X.dot(W)
  # print(scores.shape)
  # print('max before and after')
  # print(np.max(scores, axis = 1).shape)
  # print(np.max(scores, axis = 1).reshape(-1,1).shape)
  # print('max before and after | end')
  # shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1) ## reshape后，从(500,)到(500,1)
  # print(shift_scores.shape)
  # print('softmax before and after')
  # print(np.sum(np.exp(shift_scores), axis = 1).shape)
  # print(np.sum(np.exp(shift_scores), axis = 1).reshape(1,-1).shape)
  # print('softmax before and after | end')
  # softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  # ## np.sum(np.exp(shift_scores), axis = 1) 的 shape 从 (500,) 到 (1,500)
  # print(softmax_output.shape)
  # loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  # print(loss.shape)
  # loss /= num_train 
  # loss +=  0.5* reg * np.sum(W * W)
  # dS = softmax_output.copy()
  # dS[range(num_train), list(y)] += -1
  # dW = (X.T).dot(dS)
  # dW = dW/num_train + reg* W 

# 500
# 1176.6457051797192
# naive loss: 2.353291e+00 computed in 0.128064s
# (500, 10)
# max before and after
# (500,)
# (500, 1)
# max before and after | end
# (500, 10)
# softmax before and after
# (500,)
# (1, 500)
# softmax before and after | end
# (500, 10)
# ()
# vectorized loss: 2.353291e+00 computed in 0.004011s
# Loss difference: 0.000000
# Gradient difference: 0.000000

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

