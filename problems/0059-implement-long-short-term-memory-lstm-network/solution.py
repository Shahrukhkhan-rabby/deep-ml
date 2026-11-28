import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        x = np.asarray(x, dtype=float)
        h = np.array(initial_hidden_state, dtype=float, copy=True).reshape(self.hidden_size, 1)
        c = np.array(initial_cell_state, dtype=float, copy=True).reshape(self.hidden_size, 1)

        seq_len = x.shape[0]
        outputs = np.zeros((seq_len, self.hidden_size, 1), dtype=float)

        for t in range(seq_len):
            x_t = x[t].reshape(self.input_size, 1)
            z_t = np.vstack((h, x_t))

            f_t = self._sigmoid(self.Wf @ z_t + self.bf)
            i_t = self._sigmoid(self.Wi @ z_t + self.bi)
            c_tilde_t = np.tanh(self.Wc @ z_t + self.bc)
            o_t = self._sigmoid(self.Wo @ z_t + self.bo)

            c = f_t * c + i_t * c_tilde_t
            h = o_t * np.tanh(c)
            outputs[t] = h

        return outputs, h, c