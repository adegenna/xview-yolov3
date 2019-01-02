import numpy as np

def fcn_sigma_rejection(x,srl=3,ni=3):
    """
    Function to perform sigma rejection on a dataset.
    
    | **Inputs:**
    |    *x:* dataset
    |    *srl:* desired cutoff number of standard deviations for rejection  
    |    *ni:* desired number of iterations

    | **Outputs:**
    |    *x:* dataset with outliers removed
    |    *inliers:* indices of inliers w.r.t. original dataset
    """
    x       = x.astype(float)
    sizeX   = np.shape(x)
    x       = x.ravel()
    inliers = np.isfinite(x);
    for j in range(ni):
        newoutliers = (~inliers) & (~np.isnan(x));
        if ((j>0) & (~np.any(newoutliers))):
            break;
        sum_inliers = np.sum(inliers);
        if (sum_inliers < 3):
            break;
        x[newoutliers] = np.nan
        mu    = np.nansum(x)/sum_inliers
        xms   = (x-mu)**2
        sigma = np.sqrt(1./(sum_inliers-1)*np.nansum(xms))
        if (sigma == 0):
            break;
        inliers = (xms < (srl*sigma)**2)
    x       = x[inliers]
    inliers = np.reshape(inliers,sizeX)
    return x,inliers
