from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    pass
    
    z =  prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(z)
    cache = (x,prev_h,Wx,Wh,next_h)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    pass
    x, prev_h, Wx, Wh,  next_h = cache
    ## 上述几个变量的shape分别是 ：
    ##  (N, D)  (N, H)  (D, H)  (H, H)  (N, H)  (N, H)
    # z =  prev_h.dot(Wh) + x.dot(Wx) + b
    # next_h = np.tanh(z)
    # f(z)' = 1 − (f(z))2

    dz = dnext_h *  ( 1 - np.square(next_h)  )            ## shape : (N, H)
    db = np.sum(dz, axis = 0)  ## shape :(H)
    dx = dz.dot(Wx.T)           # Wx * dz 
    # dprev_h = dz.dot(Wh)   
    dprev_h = dz.dot(Wh.T) ## 为什么需要转置？
    dWx = (x.T).dot(dz)
    dWh = (prev_h.T).dot(dz)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    
    ### 对输入的 T 个向量进行运算
    N, T, D = x.shape 
    H = b.shape[0]

    ## 定义一个空白的h来装结果
    h = np.zeros((N, T, H))


    prev_h = h0
    for t in range(T):
        current_x = x[:,t,:]
        ## 运算
        next_h, cache_t = rnn_step_forward(current_x, prev_h, Wx, Wh, b)
        prev_h = next_h
        ## 填充h结果
        h[:,t,:] = next_h

    ## 等到最后才来存储cache？
    cache = (x, h0, Wx, Wh, b, h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    ## dh 包含了各个时刻的dh
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    pass

    ## 计算出最终的结果 dx, dh0, dWx, dWh, db
    ## 需要对每一个时间片t 求取 dx, dht, dWx, dWh, db
    ## 这里需要注意一下单个时间片计算出 dx, dht, dWx, dWh, db 所需要的变量
    ## 反复调用 rnn_step_backward(dnext_h, cache): 其中 cache = (x,prev_h,Wx,Wh,next_h)
    ## 

    N, T, H = dh.shape
    x, h0, Wx, Wh, b, h = cache 

    ## 通过以上数据构造出 最终输出的基本形态
    dx = np.zeros_like(x)
    dh0 = np.zeros((H ,H ))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    ## 存储某个时间片内反向传播得到的 dh。在最后一个时间片的情况下，其为0.
    dh_t = np.zeros((N,H))

    ## 对每一个时间片都进行计算
    for i in range(T):
        ## 从最后一个时间片开始
        t = (T - 1) - i

        ## 构造这个时间片反向传播所需的参数cache_t
        ## 参数共享，每一个时间块参与计算的参数都是一样的
        x_t = x[:,t,:]

        ### 特别注意边界 ###
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:,t-1,:]
        next_h = h[:,t,:]

        # dnext_h = dh[:,t,:]   ### ERROR!!!
        dnext_h = dh[:,t,:] + dh_t  ## 需要再  dh[:,t,:]  的基础上加上上一个时间快计算出来的dh
        cache_t = x_t, prev_h, Wx, Wh, next_h
        dx_t, dh_t, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h,cache_t)
        
        dx[:,t,:] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db  += db_t

    dh0 = dh_t
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V. ## 每个batch大小为N，其中每一个序列的长度为T 
      x 说明了在当前大小为N的minibatch下，每一个单词在词汇表中的下表

    - W: Weight matrix of shape (V, D) giving word vectors for all words.  ## 一共有 V 个词，每一个词的维度为D
      W 表示大小为V的词汇表中每一个单词的向量表示。


    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - 为一个minibatch的中的单词在词汇表中找到对应的向量表示
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    pass

    N, T = x.shape
    V, D = W.shape

    out = np.zeros((N, T, D))

    ## 遍历minibatch中的每一个单词
    for n in range(N):
        for t in range(T):
            ## 第(n,t)个单词，找出其向量表示
            out[n, t] = W[x[n, t]]

    cache = x, W
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    # Performs unbuffered in place operation on operand ‘a’                      #
    # for elements specified by ‘indices’.                                       #
    # 例子:                                                                      #             
    # Increment items 0 and 1, and increment item 2 twice:                       #                                                           
    # a = np.array([1, 2, 3, 4])                                                 #                                   
    # np.add.at(a, [0, 1, 2, 2], 1)                                              #                                   
    # print(a)                                                                   #               
    # >>> array([2, 3, 5, 4])                                                    #                               
    ##############################################################################
    pass

    x, W = cache
    N, T, _ = dout.shape
    V, D = W.shape

    dW = np.zeros_like(W)

    for n in range(N):
        for t in range(T):
            dW[x[n,t]] += dout[n,t]

    # 可以直接使用下边的函数一句话实现
    # np.add.at(dW,x,dout)  ### 对于dW, 在dW[N,T]位置，一次性加上 dout

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    pass
    
    N, D = x.shape
    _, H = prev_h.shape

    a = x.dot(Wx) + prev_h.dot(Wh)  + b # shape (N,4H)
    #即a分均等的4各部分，分别作为 i f o g 四个门的输入
    ai = a[:,:H]
    af = a[:,H:2*H]
    ao = a[:,2*H:3*H]
    ag = a[:,3*H:]

    ## 计算几个门的结果
    i = sigmoid(ai)  # shape (N, H)
    f = sigmoid(af)  # shape (N, H)
    o = sigmoid(ao)  # shape (N, H)
    g = np.tanh(ag)  # shape (N, H)

    ## 计算 ct
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = x, prev_h, prev_c, Wx, Wh, b ,i ,f, o ,g,next_c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    pass
    x, prev_h, prev_c, Wx, Wh, b ,i ,f, o ,g,next_c = cache

    # 根据next_h = o * np.tanh(next_c)，可求得 do,
    do = dnext_h * np.tanh(next_c)
    # 根据next_c = f * prev_c + i * g, 可以求df, dprev_c, di, dg
    df = prev_c * dnext_c
    dprev_c = f * dnext_c
    di =  g * dnext_c
    dg =  i * dnext_c


    dai = i * ( 1 - i )
    daf = f * ( 1 - f )
    dao = o * ( 1 - o ) ## 根据 sigmoid 的导数计算公式 f′(z) = f(z)(1−f(z))
    dag = 1 - np.tanh(g)**2

    da = np.vstack((dai,daf,dao,dag)) # shape (N, 4H)

    ## 根据  a = x.dot(Wx) + prev_h.dot(Wh) + +b 求 
    db = np.sum( da ,axis = 0)
    dx = da.dot(Wx.T)
    dWx = (x.T).dot(da)
    dprev_h = da.dot(Wh.T)  
    dWh = (prev_h.T).dot(da)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. 

    The input x gives scores for all vocabulary elements at all timesteps, 
    and y gives the indices of the ground-truth element at each timestep. 
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps 
    and averaging across the minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. 
    The optional mask argument tells us which elements should contribute to the loss. 用来屏蔽掉NULL

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
