from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
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
        # Giả sử bạn gộp input_dim, hidden_dims, và num_classes thành một list các dimensions
        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(1, self.num_layers + 1):
        # 1. Khởi tạo W và b cho layer thứ i
            self.params['W' + str(i)] = np.random.normal(0, weight_scale, (dims[i-1], dims[i]))
            self.params['b' + str(i)] = np.zeros(dims[i])

        # 2. Khởi tạo gamma và beta nếu dùng batchnorm (bỏ qua layer cuối)
            if self.normalization == 'batchnorm' and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(dims[i])
                self.params['beta' + str(i)] = np.zeros(dims[i])
            
            if self.normalization == 'layernorm' and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(dims[i])
                self.params['beta' + str(i)] = np.zeros(dims[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
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
        caches = {}
        current_out = X
        
        # --- Đi qua các hidden layers ---
        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            
            # Tùy thuộc vào việc có dùng BatchNorm hay không để gọi hàm forward tương ứng
            if self.normalization == 'batchnorm':
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                bn_param = self.bn_params[i-1]
                # Gọi hàm gộp affine_batchnorm_relu_forward
                current_out, cache = affine_batchnorm_relu_forward(current_out, W, b, gamma, beta, bn_param)
            elif self.normalization == 'layernorm':
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                ln_param = self.bn_params[i-1]

                fc_out, fc_cache = affine_forward(current_out, W, b)
                ln_out, ln_cache = layernorm_forward(fc_out, gamma, beta, ln_param)
                current_out, relu_cache = relu_forward(ln_out)

                cache = (fc_cache, ln_cache, relu_cache)
            else:
                # Gọi hàm affine_relu_forward bình thường
                current_out, cache = affine_relu_forward(current_out, W, b)
                
            caches[i] = cache
            
            # Nếu có Dropout, áp dụng ngay sau ReLU
            if self.use_dropout:
                current_out, dropout_cache = dropout_forward(current_out, self.dropout_param)
                caches['dropout' + str(i)] = dropout_cache
                
        # --- Đi qua layer cuối cùng (chỉ Affine) ---
        W_last = self.params['W' + str(self.num_layers)]
        b_last = self.params['b' + str(self.num_layers)]
        scores, cache_last = affine_forward(current_out, W_last, b_last)
        caches[self.num_layers] = cache_last
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads = 0.0, {}
    
        # 1. Tính Softmax loss và đạo hàm tại output
        data_loss, dscores = softmax_loss(scores, y)
        loss = data_loss
        
        # Cộng L2 loss cho toàn bộ W
        for i in range(1, self.num_layers + 1):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)
            
        # 2. Backprop qua layer cuối cùng (chỉ Affine)
        dout, dw, db = affine_backward(dscores, caches[self.num_layers])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db
        
        # 3. Backprop lùi dần qua các hidden layers
        for i in range(self.num_layers - 1, 0, -1):
            # Truyền lùi qua Dropout (nếu có)
            if self.use_dropout:
                dout = dropout_backward(dout, caches['dropout' + str(i)])
                
            # Truyền lùi qua cấu trúc layer
            if self.normalization == 'batchnorm':
                dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, caches[i])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta

            elif self.normalization == 'layernorm':
                fc_cache, ln_cache, relu_cache = caches[i]

                dln_out = relu_backward(dout, relu_cache)
                dfc_out, dgamma, dbeta = layernorm_backward(dln_out, ln_cache)
                dout, dw, db = affine_backward(dfc_out, fc_cache)

                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, caches[i])
                
            # Lưu dW, db và cộng gradient của L2
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
