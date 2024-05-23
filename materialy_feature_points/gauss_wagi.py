import numpy as np

def generate_gaussian_weights(size, sigma=1.0):
    """
    Generates normalized Gaussian weights for a given filter size.
    :param size: Size of the filter (e.g., 17 for a 17x17 filter).
    :param sigma: Standard deviation for the Gaussian distribution.
    :return: Normalized filter weights.
    """
    if size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")

    half_size = size // 2
    x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
    weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    weights /= np.sum(weights)  # Normalize the weights

    return weights

# Example usage for a 17x17 filter
filter_size = 3
gaussian_weights = generate_gaussian_weights(filter_size)

# Print the weights in the desired format
print("float filterWeights[{}][{}] = {{".format(filter_size, filter_size))
# for i in range (10,21):
for row in gaussian_weights:
    # print("    {", end="")
    for val in row:
        print(str(val) + ", ")
print("};")
