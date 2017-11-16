"""Functions For Exercise 8 (still doesn't have vector quantization"""

def initialize_clusters(data, k):
    """initialize the k cluster centers (the means).
    input:
        data: original data with shape (num_sample, num_feature).
        k: predefined number of clusters for the k-means algorithm.
    output:
        a numpy array with shape (k, num_feature)
    """
    seed = 162.
    indices = np.random.choice(data.shape[0],k)
    centers = data[indices]
    return centers

def build_distance_matrix(data, mu):
    """build a distance matrix.
    return
        distance matrix:
            row of the matrix represents the data point,
            column of the matrix represents the k-th cluster.
    """
    dist = sp.spatial.distance.cdist(data,mu)
    return dist

def update_kmeans_parameters(data, mu_old):
    """update the parameter of kmeans
    return:
        losses: loss of each data point with shape (num_samples, 1)
        assignments: assignments vector z with shape (num_samples, 1)
        mu: mean vector mu with shape (k, num_features)
    """
    #Initialize stuff
    z = np.zeros((data.shape[0],mu_old.shape[0]))#assignment vector
    dist = build_distance_matrix(data, mu_old)
    mu = np.zeros((mu_old.shape[0],data.shape[1]))
    
    #Start looping
    for i in range(data.shape[0]):#loop through n rows of data x
        ind = np.argmin(dist[i,:])
        z[i,ind] = 1.
    for j in range(mu_old.shape[0]):#loop over k rows of mu
        sum_ = np.sum(z[:,j])
        prod = z[:,j].T.dot(data)

        mu[j,:] = (prod/sum_)
      
    #Compute loss
    diff = data.T - mu.T.dot(z.T)
    losses = np.sum(np.square(diff))
    
    return losses, z, mu

#Image Compression Functions

def preprocess_image(original_image):
    """preprocess the image.""" 
    rows = original_image[0]
    cols = original_image[1]
    processed_image = np.reshape(rows*cols, -1)
    return processed_image

#Should have the kmean_compression function here