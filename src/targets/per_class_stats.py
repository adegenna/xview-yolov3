import numpy as np

def per_class_stats(classes,w,h):
    """
    Function to calculate statistics of target data.
    
    | **Inputs:**
    |    *classes:* target data processed/produced with the Target class
    |    *w:* image width
    |    *h:* image height

    | **Outputs:**
    |    *class_mu:* mean of target classes
    |    *class_sigma:* standard deviation of target classes
    |    *class_cov:* covariance of target classes
    """
    area         = np.log(w*h)
    aspect_ratio = np.log(w/h)
    uc           = np.unique(classes)
    n            = np.size(uc)
    class_mu     = np.zeros([n,4])
    class_sigma  = np.zeros([n,4])
    class_cov    = np.zeros([n,4,4])
    for i in range(n):
        j    = np.where(classes == uc[i])[0]
        wj   = np.log(w[j])
        hj   = np.log(h[j])
        aj   = area[j]
        arj  = aspect_ratio[j]
        data = np.vstack([wj,hj,aj,arj]).T
        class_mu[i]    = np.mean(data,axis=0)
        class_sigma[i] = np.std(data,axis=0)
        class_cov[i]   = np.cov(data.T)
    return class_mu, class_sigma, class_cov
