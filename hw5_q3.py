import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# Load data
data = scipy.io.loadmat('yalefaces.mat')
data = data['yalefaces']

# Peform PCA on dataset
# Reshape each image to a vector 
img_height = data.shape[0]
img_width = data.shape[1]
num_images = data.shape[2]
X = data.reshape(-1, num_images) # one image per column
X_centered = X - np.mean(X, axis=1, keepdims=True)

# Compute covariance matrix
S = (1 / num_images) * np.dot(X_centered, X_centered.T) 

# Eigen decompisition of covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(S)

# Sort eigenvalues and eigenvectors descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Plot sorted eigenvalues
plt.plot(eigenvalues[:50])
plt.title("Sorted Top 50 Eigenvalues")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot first 20 eigenfaces
_, subplots = plt.subplots(4, 5)
for i in range(20):
    subplot = subplots[i // 5, i % 5] # access subplot at proper place
    
    eigenface = eigenvectors[:, i].reshape(img_height, img_width)
    subplot.imshow(eigenface, cmap='gray')
    subplot.set_title(f'Eigenface {i}')
    subplot.axis('off')
plt.tight_layout()
plt.show()
