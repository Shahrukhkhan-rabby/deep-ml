import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Compute output dimensions using floor division
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    # Pad input_matrix with zeros on all four sides
    padded = np.pad(input_matrix, pad_width=padding, mode="constant", constant_values=0)

    # Allocate float output array
    output_matrix = np.zeros((output_height, output_width), dtype=float)

    # Slide the kernel over the padded input (cross-correlation: no kernel flip)
    for i in range(output_height):
        for j in range(output_width):
            r_start = i * stride
            c_start = j * stride
            window = padded[r_start : r_start + kernel_height, c_start : c_start + kernel_width]
            output_matrix[i, j] = np.sum(window * kernel)

    return output_matrix