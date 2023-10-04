import cv2
import numpy as np

# Load the image
image = cv2.imread('images.jpg', cv2.IMREAD_COLOR)

# Define parameters
window_size = 20  # Size of the local block/window
gamma_low = 0.7   # Gamma correction value for mean < user specific value
gamma_high = 1.5  # Gamma correction value for mean > user specific value
mean_threshold = 100  # User specific mean threshold

# Split the image into overlapping blocks
height, width, _ = image.shape
result = np.zeros_like(image, dtype=np.uint8)

for y in range(0, height, window_size):
    for x in range(0, width, window_size):
        # Extract the local block
        block = image[y:y+window_size, x:x+window_size]

        # Compute the mean value of the block
        block_mean = np.mean(block)

        # Apply gamma correction based on the mean value
        if block_mean < mean_threshold:
            gamma = gamma_low
        else:
            gamma = gamma_high

        # Perform gamma correction on the block
        block_corrected = np.power(block / 255.0, gamma) * 255.0

        # Place the corrected block into the result image
        result[y:y+window_size, x:x+window_size] = block_corrected.astype(np.uint8)

# Show the result
cv2.imshow('Adaptive Gamma Correction', result)
cv2.waitKey(0)
cv2.destroyAllWindows()