import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.

        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))

    def forward(self, inputs, h_prev):
        xs = {}
        hs = {}
        ys = {}
        as_ = {}

        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = inputs[t]
            as_[t] = (self.W_xh @ xs[t]) + (self.W_hh @ hs[t - 1]) + self.b_h
            hs[t] = np.tanh(as_[t])
            ys[t] = (self.W_hy @ hs[t]) + self.b_y
        
        return ys, hs, xs
    
    def backward(self, ys_grad, hs, xs, clip_value=0.1):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        batch_size = ys_grad[0].shape[1]
        dh_next = np.zeros((self.hidden_dim, batch_size))
        T = len(ys_grad)

        for t in reversed(range(T)):
            dW_hy += ys_grad[t] @ hs[t].T
            db_y += np.sum(ys_grad[t], axis=1, keepdims=True)
            dh = np.dot(self.W_hy.T, ys_grad[t]) + dh_next
            da = dh * (1 - hs[t] ** 2)
            dW_xh += da @ xs[t].T
            dW_hh += da @ hs[t - 1].T
            db_h += np.sum(da, axis=1, keepdims=True)
            dh_next = np.dot(self.W_hh.T, da)

        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)
        
        return dW_xh, dW_hh, dW_hy, db_h, db_y