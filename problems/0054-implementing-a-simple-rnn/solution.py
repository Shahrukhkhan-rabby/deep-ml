import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    h = np.asarray(initial_hidden_state, dtype=float)
    Wx_arr = np.asarray(Wx, dtype=float)
    Wh_arr = np.asarray(Wh, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    for x in input_sequence:
        x_arr = np.asarray(x, dtype=float)
        # Support row-vector multiplication (x @ Wx.T + h @ Wh.T + b)
        # and column-vector multiplication (Wx @ x + Wh @ h + b)
        if Wx_arr.shape[-1] == x_arr.shape[0] and Wh_arr.shape[-1] == h.shape[0]:
            z = Wx_arr @ x_arr + Wh_arr @ h + b_arr
        else:
            z = x_arr @ Wx_arr.T + h @ Wh_arr.T + b_arr
        h = np.tanh(z)

    return np.round(h, 4).tolist()