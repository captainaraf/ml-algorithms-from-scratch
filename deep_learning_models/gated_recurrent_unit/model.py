import numpy as np

class GRU:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_xr = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hr = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_r = np.zeros((hidden_dim, 1))

        self.W_xz = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hz = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_z = np.zeros((hidden_dim, 1))

        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs, h_prev):
        xs, hs, rs, zs, h_tildes, gated_hs = {}, {}, {}, {}, {}, {}
        hs[-1] = h_prev

        for t in range(len(inputs)):
            xs[t] = inputs[t].reshape(-1, 1)
            r_pre = np.dot(self.W_xr, xs[t]) + np.dot(self.W_hr, hs[t-1]) + self.b_r
            rs[t] = self._sigmoid(r_pre)
            z_pre = np.dot(self.W_xz, xs[t]) + np.dot(self.W_hz, hs[t-1]) + self.b_z
            zs[t] = self._sigmoid(z_pre)
            gated_hs[t] = rs[t] * hs[t-1]
            h_tilde_pre = np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, gated_hs[t]) + self.b_h
            h_tildes[t] = np.tanh(h_tilde_pre)
            hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_tildes[t]

        return hs, xs, rs, zs, h_tildes, gated_hs
    
    def backward(self, dh_inputs, hs, xs, rs, zs, h_tildes, gated_hs, clip_value=1.0):
        dW_xr, dW_hr, db_r = np.zeros_like(self.W_xr), np.zeros_like(self.W_hr), np.zeros_like(self.b_r)
        dW_xz, dW_hz, db_z = np.zeros_like(self.W_xz), np.zeros_like(self.W_hz), np.zeros_like(self.b_z)
        dW_xh, dW_hh, db_h = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.b_h)

        dh_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(len(dh_inputs))):
            dh = dh_inputs[t] + dh_next
            dz = dh * (h_tildes[t] - hs[t-1])
            dz_pre = dz * zs[t] * (1 - zs[t])
            dW_xz += np.dot(dz_pre, xs[t].T)
            dW_hz += np.dot(dz_pre, hs[t-1].T)
            db_z += dz_pre

            dh_tilde = dh * zs[t]
            dh_tilde_pre = dh_tilde * (1 - h_tildes[t] ** 2)
            dW_xh += np.dot(dh_tilde_pre, xs[t].T)
            dW_hh += np.dot(dh_tilde_pre, gated_hs[t].T)
            db_h += dh_tilde_pre

            dr = np.dot(self.W_hh.T, dh_tilde_pre) * hs[t-1]
            dr_pre = dr * rs[t] * (1 - rs[t])
            dW_xr += np.dot(dr_pre, xs[t].T)
            dW_hr += np.dot(dr_pre, hs[t-1].T)
            db_r += dr_pre

            dh_next = (np.dot(self.W_hr.T, dr_pre) + 
                       np.dot(self.W_hz.T, dz_pre) + 
                       np.dot(self.W_hh.T, dh_tilde_pre))

        for grad in [dW_xr, dW_hr, db_r, dW_xz, dW_hz, db_z, dW_xh, dW_hh, db_h]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        return {
            "dW_xr": dW_xr,
            "dW_hr": dW_hr,
            "db_r": db_r,
            "dW_xz": dW_xz,
            "dW_hz": dW_hz,
            "db_z": db_z,
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "db_h": db_h
        }