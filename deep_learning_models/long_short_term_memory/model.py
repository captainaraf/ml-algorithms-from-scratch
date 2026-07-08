import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        concat_dim = input_dim + hidden_dim
        self.W = np.random.randn(4*hidden_dim, concat_dim) * 0.01
        self.b = np.zeros((4*hidden_dim, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs, h_prev, c_prev):
        concat_cache = {}
        h_cache = {}
        c_cache = {}
        gate_preact_cache = {}
        gate_cache = {}

        h_cache[-1] = h_prev
        c_cache[-1] = c_prev

        for t in range(len(inputs)):
            concat_cache[t] = np.vstack((inputs[t], h_cache[t-1]))
            gate_preact_cache[t] = np.dot(self.W, concat_cache[t]) + self.b
            
            H = self.hidden_dim

            f_pre = gate_preact_cache[t][:H]
            i_pre = gate_preact_cache[t][H:2*H]
            c_bar_pre = gate_preact_cache[t][2*H:3*H]
            o_pre = gate_preact_cache[t][3*H:]
            
            f = self._sigmoid(f_pre)
            i = self._sigmoid(i_pre)
            c_bar = np.tanh(c_bar_pre)
            o = self._sigmoid(o_pre)

            gate_cache[t] = (f, i, c_bar, o)
            c_cache[t] = f * c_cache[t-1] + i * c_bar
            h_cache[t] = o * np.tanh(c_cache[t])

        return h_cache, c_cache, gate_cache, concat_cache, gate_preact_cache
    
    def backward(self, dh_inputs, h_cache, c_cache, gate_cache, concat_cache, gate_preact_cache, clip_value=1.0):
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        batch_size = dh_inputs[0].shape[1]

        dh_next = np.zeros((self.hidden_dim, batch_size))
        dc_next = np.zeros((self.hidden_dim, batch_size))

        T = len(dh_inputs)

        for t in reversed(range(T)):
            dh = dh_inputs[t] + dh_next
            f, i, c_bar, o = gate_cache[t]
            tanh_c = np.tanh(c_cache[t])
            do = dh * tanh_c
            do_pre = do * o * (1 - o)

            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            dc_bar = dc * i
            dc_bar_pre = dc_bar * (1 - c_bar ** 2)
            di = dc * c_bar
            di_pre = di * i * (1 - i)

            df = dc * c_cache[t-1]
            df_pre = df * f * (1 - f)

            dgate_preact = np.vstack((df_pre, di_pre, dc_bar_pre, do_pre))
            dW += np.dot(dgate_preact, concat_cache[t].T)
            db += np.sum(dgate_preact, axis=1, keepdims=True)

            dconcat = np.dot(self.W.T, dgate_preact)
            dh_next = dconcat[self.input_dim:, :]
            dc_next = dc * f


        for param in [dW, db]:
            np.clip(param, -clip_value, clip_value, out=param)

        return dW, db