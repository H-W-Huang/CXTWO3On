# -*- coding: UTF-8 -*-
from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #pass
    N = x.shape[0]
    # print(x.shape)
    D = np.prod(x.shape[1:])
    # print(D)
    # print(x.shape)
    x_ = x.reshape(N,-1)
    # print(x_.shape)
    # print(w.shape)
    out = x_.dot(w) +b
    # print(out.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # print(type(x))
    N = x.shape[0]

    dx = dout.dot(w.T)
    ## 少了一部，这里的dx维度需要分散开来
    # print(">>>>>:"+str(dx.shape))
    dx = dx.reshape(*x.shape)  ## dx 的shape永远和x一致

    ## 此处还需要把x的shape 从 (N,d1,d2,...,dn)转换为(N,D)
    dw = (x.reshape(N,-1).T).dot(dout)
    db = np.sum(dout,axis=0)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x >= 0 )
    # print('>>>>>>'+str(type(out)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = (x >= 0) * dout
    # print('dx type:'+str(type(dx)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



## BN归一化实现--前向
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        ### 训练阶段
        ### 首先计算单个batch的均值和方差
        # x_mean = np.mean(x,axis=0)
        # x_var = np.var(x,axis=0)
        # ### 对每一个x进行归一化
        # x_hat = (x - x_mean) / np.sqrt( x_var + eps)

        # ### 更新 running_mean 然后 running_var
        # running_mean = momentum * running_mean + ( 1 - momentum ) * x_mean
        # running_var = momentum * running_var + ( 1 - momentum ) * x_var
        # ## 计算输出结果out
        # out = gamma * x_hat + beta
        # ## 存储所有的中间计算结果
        # cache = ( x, gamma, beta, x_hat, x_mean, x_var, eps )

        ### 按照计算图一步步计算的实现如下：
        ### 一共有 10 个步骤
        x_mean = 1./N * np.sum(x,axis = 0)
        print(x_mean.shape)
        print(x.shape)
        # x_mean = np.mean(x,axis=0)
        # print(">>>>>")
        # print(x_mean == np.mean(x,axis=0))
        x_mu = x - x_mean 
        x_mu_square = np.square(x_mu)
        x_var = 1./N * np.sum(x_mu_square,axis=0)
        x_var = np.var(x,axis=0)
        x_var_plus_eps = x_var + eps
        x_sqrt_var = np.sqrt(x_var_plus_eps)
        x_sqrt_var_i = 1. / x_sqrt_var
        # x_hat = x_mu / x_sqrt_var_i ##错了
        x_hat = x_mu * x_sqrt_var_i
        x_hat_gamma = gamma * x_hat
        # out = x_hat + beta  ## ERROR
        out = x_hat_gamma + beta
        running_mean = momentum * running_mean + ( 1 - momentum ) * x_mean
        running_var = momentum * running_var + ( 1 - momentum ) * x_var
        cache = ( x, gamma, beta, x_hat, x_mean, x_var, eps )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / np.sqrt( running_var + eps)
        out = gamma * x_hat + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    #pass
    N,D = dout.shape
    x, gamma, beta, x_hat, x_mean, x_var, eps = cache

    ### 按照计算图，从输出开始，一步步进行反向传播
    # x_mean = 1.0/N * np.sum(x,axis = 0)
    x_mu = x - x_mean 
    # x_mu_square = np.square(x_mu)
    # x_var = 1.0/N * np.sum(x_mu_square,axis=0)
    x_var_plus_eps = x_var + eps
    x_sqrt_var = np.sqrt(x_var_plus_eps)
    x_sqrt_var_i = 1.0 / x_sqrt_var
    # x_hat = x_mu / x_sqrt_var_i
    # x_hat_gamma = gamma * x_hat
    # out = x_hat_gamma + beta    

    ## 已知 dout的shape为 (N, D) 
    # gamma: Scale parameter of shape (D,)
    # beta: Shift paremeter of shape (D,)
    ## 均值和方差的shape也是 (D,)


    dx_hat_gamma = dout  # (N, D) 
    dbeta = np.sum(dout,axis = 0) ## 需要保持shape的一致, (D,) 
    # dgamma = np.sum(dx_hat_gamma.T * x_hat,axis = 0)
    dgamma = np.sum(dx_hat_gamma * x_hat,axis = 0)  ## 不需要转置？ 不需要，这里是逐元素相乘，我们希望的结果也是（N, D)
    # print("dx_hat_gamma.shape:"+str(dx_hat_gamma.shape))
    # print("x_hat.shape:"+str(x_hat.shape))
    dx_hat = dx_hat_gamma * gamma
    dx_mu1 = dx_hat * x_sqrt_var_i  ### 来自 x_hat 
    dx_sqrt_var_i = np.sum(dx_hat * x_mu,axis = 0)
    dx_sqrt_var = dx_sqrt_var_i * (-1.0) / np.square(x_sqrt_var)
    dx_var_plus_eps = 0.5 * 1.0 / np.sqrt(x_var_plus_eps) * dx_sqrt_var
    dx_var = dx_var_plus_eps
    deps = dx_var_plus_eps
    dx_mu_square = 1.0/N * np.ones((N,D))  *  dx_var
    dx_mu2 = 2 * x_mu * dx_mu_square
    ## 注意均值的shape
    # dx_mean = (-1) * (dx_mu1 + dx_mu2)
    dx_mean = (-1) * np.sum((dx_mu1 + dx_mu2),axis = 0)
    dx_1 = (dx_mu1 + dx_mu2)  ## x_mu = x - x_mean 
    # dx_2 = 1.0 / N * dx_mean  ## 这样子出来的shape就仅仅是 (D,)
    dx_2 = 1.0 / N * np.ones((N,D)) * dx_mean
    dx = dx_1 + dx_2 
 
    ## 注意累加计算咋在求导是的处理
    ## 以上代码参考自： https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # pass
    x, gamma, beta, x_hat, x_mean, x_var, eps = cache
    N,D = dout.shape
    # x_mean = 1.0/N * np.sum(x,axis = 0)
    # x_mu = x - x_mean 
    # x_mu_square = np.square(x_mu)
    # x_var = 1.0/N * np.sum(x_mu_square,axis=0)
    # x_var_plus_eps = x_var + eps
    # x_sqrt_var = np.sqrt(x_var_plus_eps)
    # x_sqrt_var_i = 1.0 / x_sqrt_var
    # x_hat = x_mu / x_sqrt_var_i
    # x_hat_gamma = gamma * x_hat
    # out = x_hat_gamma + beta    
    ## 合并步骤
    # dx_hat_gamma = dout  # (N, D) 
    # dbeta = np.sum(dout,axis = 0) 
    # # dgamma = np.sum(dx_hat_gamma.T * x_hat,axis = 0)
    # dgamma = np.sum(dx_hat_gamma * x_hat,axis = 0)  
    # dx_hat = dx_hat_gamma * gamma
    # dx_mu1 = dx_hat * x_sqrt_var_i  
    # dx_sqrt_var_i = np.sum(dx_hat * x_mu,axis = 0)
    # dx_sqrt_var = dx_sqrt_var_i * (-1.0) / np.square(x_sqrt_var)
    # dx_var_plus_eps = 0.5 * 1.0 / np.sqrt(x_var_plus_eps) * dx_sqrt_var
    # dx_var = dx_var_plus_eps
    # deps = dx_var_plus_eps
    # dx_mu_square = 1.0/N * np.ones((N,D))  *  dx_var
    # dx_mu2 = 2 * x_mu * dx_mu_square
    # # dx_mean = (-1) * (dx_mu1 + dx_mu2)
    # dx_mean = (-1) * np.sum((dx_mu1 + dx_mu2),axis = 0)
    # dx_1 = (dx_mu1 + dx_mu2)  ## x_mu = x - x_mean 
    # # dx_2 = 1.0 / N * dx_mean  
    # dx_2 = 1.0 / N * np.ones((N,D)) * dx_mean
    # dx = dx_1 + dx_2 

    dgamma = np.sum(dout * x_hat,axis = 0)
    dbeta = np.sum(dout,axis = 0)

    dx_hat =  dout * gamma
    dx_var = 0.5 * 1.0 / np.sqrt(x_var + eps) * np.sum(dx_hat * (x - x_mean) ,axis = 0) * (-1.0) / np.square(np.sqrt(x_var + eps))
    dx_mean =  (-1) * np.sum((dx_hat * (1.0 / np.sqrt(x_var + eps)) + 2 * (x - x_mean) * 1.0/N * np.ones((N,D)) *  dx_var),axis = 0)
    dx = (dx_hat * (1.0 / np.sqrt(x_var + eps))  + 2 * ( x - x_mean ) *  1.0/N * np.ones((N,D))  *  dx_var) +  1.0 / N * dx_mean 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


### Layer normalzation 正确性未知
def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    pass

    ### 和BN不同，此时不需要对训练和测试阶段做不同处理
    ### 两个阶段的操作是一样的
    ## 对整个层所有参数计算 均值
    x_mean = np.mean(x)
    x_var = np.var(x)

    x_hat = (x - x_mean) / np.sqrt(x_var + eps) ## 正则化完成

    out = gamma * x_hat + beta
    cache = (x, gamma, beta, x_mean, x_var, x_hat, eps )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass

    N,D = dout.shape
    x, gamma, beta, x_mean, x_var, x_hat, eps  = cache 

    dgamma = np.sum(dout * x_hat,axis = 0)
    dbeta = np.sum(dout,axis = 0)

    dx_hat =  dout * gamma
    dx_var = 0.5 * 1.0 / np.sqrt(x_var + eps) * np.sum(dx_hat * (x - x_mean) ,axis = 0) * (-1.0) / np.square(np.sqrt(x_var + eps))
    dx_mean =  (-1) * np.sum((dx_hat * (1.0 / np.sqrt(x_var + eps)) + 2 * (x - x_mean) * 1.0/N * np.ones((N,D)) *  dx_var),axis = 0)
    dx = (dx_hat * (1.0 / np.sqrt(x_var + eps))  + 2 * ( x - x_mean ) *  1.0/N * np.ones((N,D))  *  dx_var) +  1.0 / N * dx_mean 


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # pass
        # 根据输入数据的shape生成对应shape的掩码
        mask =  ( np.random.rand(*x.shape) < p ) / p
        out = x * mask 

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # pass
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass

    ## 读取数据
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape ## w的C和x的C是一样的
    P , S = conv_param['pad'], conv_param['stride']
    # 根据公式计算输出结果的shape
    out_W = (W - WW + 2*P) / S + 1
    out_H = (H - HH + 2*P) / S + 1
    ## 构造输出结果的结构
    ## 一张图卷积成 (F,out_H,out_W)的shape，N张就有N个这样的shape
    out = np.zeros((N,F,out_H,out_W))

    ##  卷积运算
    ### 首先根据pad的数值对输入x进行填充
    ### np.pad(array, pad_width, mode, **kwargs)
    ### pad_width : {sequence, array_like, int}
    ### Number of values padded to the edges of each axis.
    ### ((before_1, after_1), ... (before_N, after_N)) unique pad widths
    ### for each axis.
    ### ((before, after),) yields same before and after pad for each axis.
    ### (pad,) or int is a shortcut for before = after = pad width for all axes.
    ### 此处输入有四个维度，需要分别为四个维度指定pad的参数
    ###  (N, F, H', W') 
    ### N  ： 不需要
    ### F  ： 不需要
    ### H' ： 填充pad个像素，各个轴填充的数目一致
    ### W' ： 填充pad个像素，各个轴填充的数目一致
    x_padded = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode='constant')  ## 默认填充值为0

    ##  开始运算
    ### 可能这里确实需要先H后W，否则图片会被翻转
    for i in xrange(out_H):
        for j in xrange(out_W):
            ## 从 x_pad 中取出需要参与计算的部分
            ## 第一个维度：全取
            ## 第二个维度：全取
            ## 第三个维度：参与计算部分的H上的起始位置和终止位置,根据步长和卷积核大小确定；
            ## 第四个维度：参与计算部分的W上的起始位置和终止位置,根据步长和卷积核大小确定；
            x_padded_computed_h_start = i * S 
            x_padded_computed_h_end = i * S + HH
            x_padded_computed_w_start = j * S 
            x_padded_computed_w_end = j * S + WW
            x_padded_computed = x_padded[:, :, x_padded_computed_h_start:x_padded_computed_h_end ,x_padded_computed_w_start:x_padded_computed_w_end]
            ## 一共有F个卷积核，需要计算F次;对N张图都做这样的运算
            for f in xrange(F):
                ## 单一一张图的计算计算结果为：
                ### np.sum(x_padded_computed[i] * w[f,:,:,:])
                result = x_padded_computed *  w[f,:,:,:]
                out[:,f,i,j] = np.sum(result,axis=(1,2,3)) ## 对N中的每一个n，计算得到k,i,j位置的数值

    ## 最后对计算结果加上偏置项
    ## 需要先转换b的格式，以便广播
    ## b[None,:,None,None]
    ## 举个例子：
    ## In [51]: b = np.array([1,2,3])
    ## In [52]: b.shape
    ## Out[52]: (3L,)
    ## In [53]: b[None,:,None,None]
    ## Out[53]:
    ## array([[[[1]],
    ## 
    ##         [[2]],
    ## 
    ##         [[3]]]])
    ## In [54]: b[None,:,None,None].shape
    ## Out[54]: (1L, 3L, 1L, 1L)
    ## In [55]: b.reshape(3,-1)
    ## Out[55]:
    ## array([[1],
    ##        [2],
    ##        [3]])
    ## In [56]: b.reshape(3,-1,-1)
    out = out + (b)[None,:,None,None]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass

    x, w, b, conv_param = cache
    ## 整个过程和前向传播类似，同样需要借助x_padded来完成操作
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape ## w的C和x的C是一样的
    P , S = conv_param['pad'], conv_param['stride']
    # 根据公式计算输出结果的shape
    out_W = (W - WW + 2 * P) / S + 1
    out_H = (H - HH + 2 * P) / S + 1

    x_padded = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode='constant')
    ## 同时，还要确定好输出的shape
    ## 输出分别为  dx, dw, db
    ## dx 和 x 的shape相同， dw 和 w 的shape相同， db 和 b 的shape相同
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx_padded = np.zeros_like(x_padded)  
    ## 最终的 dx 需要借助 dx_padded来实现

    # (b)[None,:,None,None]
    ## db = dout，不过shape要一致，通过sum来实现
    db = np.sum(dout,axis = (0,2,3))


    for i in xrange(out_H):
        for j in xrange(out_W):
            x_padded_computed = x_padded[:, :, i * S :i * S + HH,j * S:j * S + WW ]
            ## 计算dw
            for f in xrange(F):
                ### dw 基础运算为 dout * x
                result = x_padded_computed * (dout[:,f,i,j])[:,None,None,None]
                dw[f,:,:,:] += np.sum(result, axis = 0)   ## 轴？？
            ## 计算dx_padded
            for n in xrange(N):
                ### 基础运算为 dout * w 
                ### w:(F, C, HH, WW)
                result = w[:,:,:,:] * (dout[n, :, i, j])[:,None,None,None]  ##这样转换？
                dx_padded[n, :, i*S:i*S+HH, j*S:j*S+WW ] += np.sum(result,axis=0)

    dx = dx_padded[:,:,P:-P,P:-P]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pass

    ## 计算上和卷积层的计算类似
    N, C, H, W = x.shape
    pool_height , pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    out_H = 1 + (H - pool_height) / stride
    out_W = 1 + (H - pool_width) / stride

    ## 构建输出结果的结构
    out = np.zeros((N,C,out_H,out_W))

    ## 开始池化
    for i in xrange(out_H):
        for j in xrange(out_W):
            # print('h:'+str(i*stride+pool_height))
            # print('w:'+str(j*stride+pool_width))
            out_computed = x[:,:, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            # print(out_computed.shape)
            ## 在 H 和 W范围内取出max，所以使用的维度是2,3
            result = np.max(out_computed,axis=(2,3))

            out[:,:,i,j] = result



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    pass
    
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height , pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    out_H = 1 + (H - pool_height) / stride
    out_W = 1 + (H - pool_width) / stride


    dx = np.zeros_like(x)

    for i in xrange(out_H):
        for j in xrange(out_W):
            ## dout[:,:,i,j] 在x上对应的区域
            x_computed = x[:,:, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            ## dout[:,:,i,j] 在dx上对应的区域
            dx_computed = dx[:,:, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]

            # 借助x_computed得到一个关于max值的Ture/False的array
            ## 同样是关于2,3轴取的最大值
            # print(np.max(x_computed, axis=(2,3)).shape)
            # print(np.max(x_computed, axis=(2,3),keepdims=True).shape)
            # (3L, 2L)
            # (3L, 2L, 1L, 1L)
            flags = np.max(x_computed, axis=(2,3),keepdims=True) == x_computed  ## 未使用keepdims导致计算错误，误差为1.0
            # print(dout)
            # print((dout[:,:,i,j])[:,:,None,None].shape) ## (3L, 2L, 1L, 1L)
            # print(flags.shape)  ## (3L, 2L, 2L, 2L) , x = np.random.randn(3, 2, 8, 8)
            # print((flags* (dout[:,:,i,j])[:,:,None,None]).shape)   # shape (3L, 2L, 2L, 2L)
            dx_computed += flags* (dout[:,:,i,j])[:,:,None,None]  ## 最大的那一个数的梯度是打开着的，其他的关闭

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass

    ## 2维以上的列表，transpose作用如下L:
    # In [17]: x.transpose(0,1,2).shape
    # Out[17]: (10L, 2L, 3L)
    # In [18]: x.transpose(1,0,2).shape
    # Out[18]: (2L, 10L, 3L)
    ## 将C分离到最后去
    N, C, H, W = x.shape
    BN_input = x.transpose(0,2,3,1)
    ## 合并前三轴的维度
    BN_input = BN_input.reshape(N*H*W, C)
    out, cache =  batchnorm_forward(BN_input, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C)
    out = out.transpose(0,3,1,2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    N, C, H, W = dout.shape
    BN_dout = dout.transpose(0,2,3,1)
    ## 合并前三轴的维度
    BN_dout = BN_dout.reshape(N*H*W, C)
    dx, dgamma, dbeta =  batchnorm_backward(BN_dout,cache)
    dx = dx.reshape(N, H, W, C)
    dx = dx.transpose(0,3,1,2)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ## 参考 https://www.leiphone.com/news/201803/UMVcCCcin2yPxu7t.html
    N, C, H, W = x.shape
    ## 对同一个通道上的数据点分组   
    x = x.reshape(N,G,C//G,H,W)  # // 表示整数除,不四舍五入

    ## 计算均值
    x_mean = np.mean(x,axis=(0,1))
    print(x_mean.shape)
    # print(x.shape)
    x_var = np.var(x,axis=(0,1))

    x_hat = x - x_mean / np.sqrt(x_var + eps)

    x_hat = x_hat.reshape(N, C, H, W)
    out = gamma * x_hat + beta
    cache = x, x_mean, x_var, x_hat, gamma, beta ,eps, G
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass

    N, C, H, W = dout.shape
    x, x_mean, x_var, x_hat, gamma, beta ,eps, G = cache

    dgamma = np.sum(dout * x_hat,axis = 0)
    dbeta = np.sum(dout,axis = 0)

    ## 计算完dgamma和dbeta后在进行转换
    dx_hat =  dout * gamma

    #dout = dout.reshape(N,G,C//G,H,W)
    x = x.reshape(N,G,C//G,H,W)
    print(x.shape)
    print(x_mean.shape)
    print(x_var.shape)
    print(dx_hat.shape)
    dx_hat = dx_hat.reshape(N,G,C//G,H,W)
    dx_var = 0.5 * 1.0 / np.sqrt(x_var + eps) * np.sum(dx_hat * (x - x_mean) ,axis = 0) * (-1.0) / np.square(np.sqrt(x_var + eps))
    dx_mean =  (-1) * np.sum((dx_hat * (1.0 / np.sqrt(x_var + eps)) + 2 * (x - x_mean) * 1.0/N * np.ones((N,G,C//G,H,W)) *  dx_var),axis = 0)
    dx = (dx_hat * (1.0 / np.sqrt(x_var + eps))  + 2 * ( x - x_mean ) *  1.0/N * np.ones((N,G,C//G,H,W))  *  dx_var) +  1.0 / N * dx_mean 
    
    dx = dx.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
