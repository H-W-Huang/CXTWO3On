# -*- coding: UTF-8 -*-
from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        #pass

        ## 初始化
        D = input_dim
        H = hidden_dim
        C = num_classes
        # self.params['W1'] = np.random.normal(0.0,weight_scale,(D,H))
        self.params['W1'] = weight_scale *  np.random.randn(D,H)
        self.params['b1'] = np.zeros(H)  ## 此处不是D而是 H
        # self.params['W2'] = np.random.normal(0.0,weight_scale,(H,C))
        self.params['W2'] = weight_scale *  np.random.randn(H,C)
        self.params['b2'] = np.zeros(C)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #pass
        ### 首先是前向传播
        z1, cache1_1 = affine_forward(X,self.params['W1'],self.params['b1'])
        a1, cache1_2 = relu_forward(z1)
        z2, cache2_1 = affine_forward(a1,self.params['W2'],self.params['b2'])
        scores = z2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        loss, dout = softmax_loss(z2, y)
        ## 开始反向传播
        ## 各个梯度的命名依据正向传播的命名
        da1,dw2,db2 = affine_backward(dout,cache2_1)  ## out, cache2_1 = affine_forward(a1,self.params['W2'],self.params['b2'])
        dz1 = relu_backward(da1,cache1_2)             #  a1, cache1_2 = relu_forward(z1)
        dx,dw1,db1  = affine_backward(dz1,cache1_1)   #  z1, cache1_1 = affine_forward(X,self.params['W1'],self.params['b1'])


        # 最终的 loss 和 梯度 还需要进行正则化
        grads['W2'] = dw2 +  self.reg * self.params['W2'] 
        grads['b2'] = db2
        grads['W1'] = dw1 +  self.reg * self.params['W1'] 
        grads['b1'] = db1 
        loss = loss + 0.5 * self.reg * ( np.sum( self.params['W1'] * self.params['W1'] ) +  np.sum( self.params['W2'] * self.params['W2'] ) )
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #pass

        ### 初始化参数
        D = input_dim
        C = num_classes

        hidden_nums = len(hidden_dims)
        first_dim = D
        second_dim = None


        for i in xrange(len(hidden_dims)):
            param_w_name = 'W'+str(i+1) 
            param_b_name = 'b'+str(i+1) 
            second_dim = hidden_dims[i]
            # self.params[param_w_name] = weight_scale * np.random.rand(first_dim,second_dim)  ## ERROR! rand -> randn
            self.params[param_w_name] = weight_scale * np.random.randn(first_dim,second_dim)
            self.params[param_b_name] = np.zeros(second_dim)
            ## 如果使用了BN，还需要对BN的参数进行初始化
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                self.params['gamma%d'%(i+1)] = np.ones(second_dim)
                self.params['beta%d'%(i+1)] = np.zeros(second_dim)
            first_dim = second_dim

        ## 最后的连接层和输出层直接的参数
        param_w_name = 'W'+str(len(hidden_dims)+1)
        param_b_name = 'b'+str(len(hidden_dims)+1)
        self.params[param_w_name] = weight_scale * np.random.randn(first_dim,C)
        self.params[param_b_name] = np.zeros(C)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        ## 目前没有实现dropout和BN的情况

        current_input = X
        caches = {}  
        ## 有 self.num_layers 参数可用
        ## 则遍历所有的 hidden_layers,以层为单位来思考整个过程
        for i in xrange(self.num_layers - 1):
            # notebook中的例子，层数为3
            # print('processing '+str(i))
            current_W = self.params['W'+str(i+1)]
            current_b = self.params['b'+str(i+1)]
            out = None
            cache = None
            # 分支, 是否使用 BN
            if self.normalization=='batchnorm':
                ## 使用BN
                ## affine
                out1, cache1 = affine_forward(current_input,current_W,current_b)
                ## bn
                out2, cache2 = batchnorm_forward(out1,self.params['gamma%d'%(i+1)],self.params['beta%d'%(i+1)],self.bn_params[i])
                ## relu
                out,cache3 = relu_forward(out2)
                cache = (cache1,cache2,cache3)
                # print(np.array(cache1).shape)
                # print(cache[1].shape)
                # print(cache[2].shape)                
            elif self.normalization == 'layernorm':
                out1, cache1 = affine_forward(current_input,current_W,current_b)
                ## bn
                out2, cache2 = layernorm_forward(out1,self.params['gamma%d'%(i+1)],self.params['beta%d'%(i+1)],self.bn_params[i])
                ## relu
                out,cache3 = relu_forward(out2)
                cache = (cache1,cache2,cache3)
            else:
                ## 正向传播，同时顺带计算relu的结果
                out,cache = affine_relu_forward(current_input,current_W,current_b)

            ## 是否使用Dropout
            if self.use_dropout :
                # print('using dropout:'+ str(self.dropout_param))
                out,dropout_cache = dropout_forward(out,self.dropout_param) ## 整合dropout的cache
                cache = (dropout_cache,cache)
            current_input = out
            caches[i] = cache
            # 此处最终只有caches[0]和caches[1]

        ## 计算最后的输出
        out,cache = affine_forward(
                        current_input,
                        self.params['W'+str(self.num_layers)],
                        self.params['b'+str(self.num_layers)]
                    )

        ### 这里和上边的最后一个编号之间可能不连续
        caches[self.num_layers ] = cache
        scores = out


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass

        ## 首先是计算损失
        loss, dout = softmax_loss(scores,y)

        loss = loss + 0.5 * self.reg * np.sum( self.params['W%d'%(self.num_layers)] * self.params['W%d'%(self.num_layers)] )
        # print('>>>>>current_index:'+str(self.num_layers))

        ## 反向传播和正向传播一样，分离出最后的HiddenLayer和OutputLayer之间的计算，其他层间的在一个循环里完成
        ### 首先是输出层和最后一个隐层之间的梯度
        current_dx, dw, db = affine_backward(dout, caches[self.num_layers])
        grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db

        for i in xrange(self.num_layers -1):
            current_index = self.num_layers -1 -i
            loss = loss + 0.5 * self.reg * np.sum( self.params['W%d'%(current_index)] * self.params['W%d'%(current_index)] )
            caches_being_used = caches[current_index-1]

            # current_dx, dw, db =  None
            ## 开始分支
            ## 首先是dropout
            if self.use_dropout:
                dropout_cache,caches_being_used = caches_being_used
                current_dx = dropout_backward(current_dx,dropout_cache) ## 分离出dropout的cache
                # print('dropout backward finish!')
            if self.normalization == 'batchnorm':
                cache = caches_being_used
                current_dx  = relu_backward(current_dx, cache[2])
                current_dx, dgamma, dbeta = batchnorm_backward(current_dx, cache[1])
                current_dx,dw,db = affine_backward(current_dx, cache[0])
            elif self.normalization == 'layernorm':
                cache = caches_being_used
                current_dx  = relu_backward(current_dx, cache[2])
                current_dx, dgamma, dbeta = layernorm_backward(current_dx, cache[1])
                current_dx,dw,db = affine_backward(current_dx, cache[0])
            else:
                fc_cache, relu_cache = caches_being_used
                current_dx = relu_backward(current_dx, relu_cache)
                # print('>>>'+str(fc_cache))
                current_dx, dw, db = affine_backward(current_dx, fc_cache)

                # current_dx, dw, db = affine_relu_backward(current_dx, caches_being_used)
            
            grads['W%d'%(current_index)] = dw + self.reg * self.params['W%d'%(current_index)]
            grads['b%d'%(current_index)] = db

            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                grads['gamma%d'%(current_index)] = dgamma
                grads['beta%d'%(current_index)]  = dbeta
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
