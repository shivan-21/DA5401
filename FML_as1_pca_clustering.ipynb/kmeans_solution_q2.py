#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # USED ONLY FOR DATA LOADING


# ## Load and Visualise the Data

# In[71]:


# Load the dataset
dataset2_path = r'C:\Users\shiva\OneDrive\Desktop\DA5401_Big_Data_Lab\FML_as1_pca_clustering.ipynb\dataset1-assignment1 - Sheet1.csv'
X2 = pd.read_csv(dataset2_path, header=None).values

print("Shape, minimum value, maximum value:\n", X2.shape, X2.min(), X2.max())
print("First 5 rows:\n", X2[:5, :])


# In[72]:


# Visualize  original data
plt.figure(figsize=(8, 6))
plt.scatter(X2[:, 0], X2[:, 1], c='teal', alpha=0.6)
plt.title('Original Data (Dataset 2)')
plt.xlabel('Feature 1, i.e. column 1',)
plt.ylabel('Feature 2, i.e. column 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# # Implement Kmeans Clustering from Scratch
# Implement K-means from scratch and run it with k=4 for 5 different random initializations.
# 

# In[73]:


class KMeans:
    """K-means clustering implementation from scratch."""
    def __init__(self, k=4, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.error_history = []

    def _initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        self.centroids = X[random_idx[:self.k]]

    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        return new_centroids

    def _compute_error(self, X, labels):
        error = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                error += np.sum((cluster_points - self.centroids[i])**2)
        return error

    def fit(self, X):
        self.error_history = []
        self._initialize_centroids(X)
        for i in range(self.max_iters):
            labels = self._assign_clusters(X)
            self.error_history.append(self._compute_error(X, labels))
            new_centroids = self._update_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self._assign_clusters(X)


# ## Run K means
# - plots error vs iteration graph

# In[74]:


print("Running K-means with k=4 for 5 different initializations:")
num_runs = 5
k = 4 # no. of centroids, i.e. clusterss

for run in range(num_runs):
    print(f"Run index:  {run+1}/{num_runs}")
    kmeans = KMeans(k=k, max_iters=100, random_state=run)
    labels = kmeans.fit(X2)
    plt.figure(figsize=(12, 5))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot WCSS for the run 
    ax[0].plot(range(1, len(kmeans.error_history) + 1), kmeans.error_history, marker='o')
    ax[0].set_title(f'Run {run+1}: Error (WCSS) vs. Iteration')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('WCSS Error')
    ax[0].grid(True)

    # Plot final clusters with centroids
    scatter = ax[1].scatter(X2[:, 0], X2[:, 1], c=labels, cmap='winter', alpha=0.8)
    ax[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    ax[1].set_title(f'Run {run+1}: Final Clusters (k={k})')
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 2')
    ax[1].legend(*scatter.legend_elements(), title='Clusters')
    ax[1].grid(True)



# ## Plot Voronoi Regions
# 
# Run K-means for K = {2, 3, 4, 5} with a fixed initialization and
# plot the Voronoi regions.
# 

# In[85]:


# helper function to plot Voronoi regions

def plot_voronoi(X, kmeans_model, k):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    temp_model = KMeans(k=k)
    temp_model.centroids = kmeans_model.centroids
    Z = temp_model._assign_clusters(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='magma')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans_model._assign_clusters(X), s=20, cmap='magma', edgecolor='k')
    plt.scatter(kmeans_model.centroids[:, 0], kmeans_model.centroids[:, 1], c='red', marker='x', s=200)
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.title(f'K-means Clustering and Voronoi Regions (K={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# In[86]:


print("Plots of Voronoi Regions for different K values:")
K_values = [2, 3, 4, 5]
# use random state 42 as the fixed seed 
fixed_seed = 42

for k_val in K_values:
    print(f"Running for K={k_val}")
    kmeans_voronoi = KMeans(k=k_val, random_state=fixed_seed)
    kmeans_voronoi.fit(X2)
    plot_voronoi(X2, kmeans_voronoi, k=k_val)


# # Spectral Clustering 
# 
# Implement spectral clustering, which can find non-convex clusters

# In[78]:


# kernel helper functions

def rbf_kernel(x1, x2, sigma=1.0):
    sq_dists = -2 * (x1 @ x2.T) + np.sum(x1**2, axis=1)[:, np.newaxis] + np.sum(x2**2, axis=1)
    return np.exp(-sq_dists / (2 * sigma**2))

def polynomial_kernel(x1, x2, degree=2, c=1):
    return (x1 @ x2.T + c) ** degree


# In[79]:


class SpectralClustering_rbf:

    '''Spectral Clustering using RBF (Gaussian) Kernel.
    Uses the previously defined rbf_kernel function to compute the similarity matrix.
    '''
    def __init__(self, k=4, sigma=1.0, random_state=None):
        self.k = k
        self.sigma = sigma
        self.random_state = random_state

    def fit_predict(self, X):
        W = rbf_kernel(X, X, sigma=self.sigma)
        eigenvalues, eigenvectors = np.linalg.eig(W)
        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvectors = eigenvectors[:, idx]
        V = eigenvectors[:, :self.k].real
        kmeans = KMeans(k=self.k, random_state=self.random_state)
        return kmeans.fit(V)

class SpectralClustering_poly:
    ''' Spectral Clustering using Polynomial Kernel.
    Uses the previously defined polynomial_kernel function to compute the similarity matrix.'''
    def __init__(self, k=4, degree=2, c=1, random_state=None):
        self.k = k
        self.degree = degree
        self.c = c
        self.random_state = random_state

    def fit_predict(self, X):
        W = polynomial_kernel(X, X, degree=self.degree, c=self.c)
        eigenvalues, eigenvectors = np.linalg.eig(W)
        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvectors = eigenvectors[:, idx]
        V = eigenvectors[:, :self.k].real
        kmeans = KMeans(k=self.k, random_state=self.random_state)
        return kmeans.fit(V)


# ### run spectral clustering using thte RBF kernel

# In[91]:


print("Running Gaussian Spectral Clustering with k=4")
sigma_choices = [0.02, 0.5, 1.0, 1.5, 5]

for sigma in sigma_choices: # This may need tuning based on the data's scale
    spectral = SpectralClustering_rbf(k=4, sigma= sigma, random_state=42)
    labels_spectral = spectral.fit_predict(X2)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c= labels_spectral, cmap='rainbow', alpha=0.6, edgecolors='k')
    plt.title(f'Spectral Clustering Results (k=4, σ={sigma})')
    plt.xlabel('Feature 1')
    plt.legend(*scatter.legend_elements(), title='Clusters (RBF Kernel)', loc='upper right')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()


    # plt.scatter(X[:, 0], X[:, 1], c=kmeans_model._assign_clusters(X), s=20, cmap='magma', edgecolor='k')
    # plt.scatter(kmeans_model.centroids[:, 0], kmeans_model.centroids[:, 1], c='red', marker='x', s=200)
    # plt.legend(*scatter.legend_elements(), title='Clusters')


# ### run polynomial Spectral Clustering 

# In[103]:


print("Running Polynomial Spectral Clustering with k=4")
degree_list = [3, 5,10, 50]

poly_c = 1

for deg in degree_list:
    spectral_poly = SpectralClustering_poly(k=4, degree= deg, c=poly_c, random_state=42)
    labels_spectral_poly = spectral_poly.fit_predict(X2)


    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels_spectral, cmap='Spectral', alpha=0.6, edgecolors='k')
    plt.title(f'Spectral Clustering Results (k=4, polynomial degree={deg}, c={poly_c})')
    plt.xlabel('Feature 1')
    plt.legend(*scatter.legend_elements(), title='Clusters (Poly Kernel)', loc='upper right')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()


# # Spectral Clustering Argmax

# In[82]:


class SpectralClustering_argmax:
    def __init__(self, k, kernel='rbf', sigma=1.0, degree=3, c=1, random_state=None):
        self.k = k
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.c = c
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def _kernel_matrix(self, X):
        if self.kernel == 'rbf':
            # Calculate squared Euclidean distances 
            sq_dists = -2 * (X @ X.T) + np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)
            # Apply the RBF kernel formula using sigma
            return np.exp(-sq_dists / (2 * self.sigma**2))
        elif self.kernel == 'poly':
            return (X @ X.T + self.c) ** self.degree
        else:
            raise ValueError("Unsupported kernel type")

    def fit_predict(self, X):
        # 1. Compute the Kernel Matrix
        K = self._kernel_matrix(X)

        # 2. Compute Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 3. Select the top k eigenvectors
        V = eigenvectors[:, :self.k]

        # 4. Assign clusters using argmax
        labels = np.argmax(V, axis=1)
        return labels


# In[104]:


# --- Running the new Spectral Clustering ---
print("Running Spectral Clustering with argmax mapping, k=4, and sigma=0.5")
# experiment with different values of sigma

sigma_range = [0.1, 0.5, 1.0, 1.5 ,5]

for sigma in sigma_range:
    print(f"Using sigma = {sigma}")
    spectral_argmax = SpectralClustering_argmax(k=4, kernel='rbf', sigma= sigma, random_state=42)
    labels_spectral_argmax = spectral_argmax.fit_predict(X2)

    # --- Plotting the results ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels_spectral_argmax, cmap='viridis', alpha=0.8, edgecolors='k')
    plt.title(f'Spectral Clustering with Argmax Mapping (k=4, sigma= {sigma})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(*scatter.legend_elements(), title='Clusters (Argmax RBF K)', loc='upper right')
    plt.grid(True)
    plt.show()




