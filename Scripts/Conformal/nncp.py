import numpy as np

class NNCP():
    """Template for Nearest Neighbor Conformal Prediction Class."""
    def __init__(self, points, weights=None, workers=-1):
        self.points = points
        self.N = len(self.points)
        self.workers = workers

        if weights is not None:        
            self.weights = weights
        else:
            self.weights = 1/(self.N+1) * np.ones(self.N+1)
        
        self.compute_alphas()
                
        # Add infinity to alphas for case where requested confidence exceeds
        # N/(N+1)
        self.alphas = np.append(self.alphas, np.inf)
                
        self.ordering = np.argsort(self.alphas)
        self.ordered_alphas = self.alphas[self.ordering]
        
        self.init_weight_info()
            
    def init_weight_info(self):
        """A function to initialize all derivative values of the weights so
        that can easily redefine weighting if desired."""
        self.ord_weights = self.weights[self.ordering]
        self.cum_ord_weights = np.cumsum(self.ord_weights)
        
    def compute_alphas(self):
        """Compute distance to nearest self-excluding neighbor among points."""
        pass
  
    def compute_scores(self, test_points):
        """Compute distance to nearest point in self.points for test points."""
        pass
    
    def compute_cutoff(self, eps):
        """Get the alpha cutoff for defining CP set with miscoverage eps."""    
        conf = 1-eps

        # Get first index where the cumulative sum of weights exceeds conf
        # Pad very slightly to avoid roundoff error
        weighted_quants = self.cum_ord_weights - conf >= 0 # -1e-14
        # Since used the argmax, in this case k is already 0 index
        k = np.argmax(weighted_quants)
        
        cutoff = self.ordered_alphas[k]
        
        # Success if cutoff < inf
        if cutoff < np.inf:
            success = True
        else:
            success = False
            print('eps too tiny to be guaranteed')
                    
        return cutoff, success
        
    def predict_p(self, test_points=None, scores=None):
        """p-value (upper bound) for hypothesis testing for each test point 
        (separately) whether it was drawn from input distribution. 
        Null hypothesis = test point drawn from input distribution."""
        # As increase eps, reject more. Find the smallest eps for which 
        # reject the null hypothesis. If this point is very "close" to the
        # inputs then will still be contained even in a tiny confidence set
        # i.e. contained even when increase eps a lot. So, will get a large
        # value. Conversely, if this point is very "far" then will have a small
        # value.
        
        # Accept when score <= alpha_(k) and conversely reject when 
        # score > alpha_(k) 
        # So, we must find the largest alpha_(k) for which score > alpha_(k)
        # Then, extract the corresponding probability i.e. k/(N+1) = conf
        # Lastly, compute eps = 1 - conf
                
        # Score each of the test points
        # Made it so can instead pass in scores to avoid recomputing
        # if not needed
        if scores is None:
            if test_points is None:
                raise ValueError('Either pass in test_points or scores')
            scores = self.compute_scores(test_points)[0]

        # Compute for each test score the largest alpha_(k) for which
        # score > alpha_(k) 
        
        # i = insert_inds[j] satisfies (using 'left')
        # ordered_alphas[i-1] < scores[j] <= ordered_alphas[i]
        # so we take alpha_(k) = ordered_alphas[i-1] so k = i-1
        insert_inds = np.searchsorted(self.ordered_alphas, scores, 'left')
        k_vals = insert_inds - 1
        
        # Ex: score > alpha for all alpha except alpha=inf then k = N-1
        # so get 1 - self.cum_ord_weights[N-1] = 1 - N/(N+1) = 1/(N+1) in the
        # unweighted case. So lowest p-value achievable is 1/(N+1).
        p_vals = 1 - self.cum_ord_weights[k_vals]
        
        # Suppose scores[j] <= alpha for all alpha i.e. always accept
        # then i = 0, k = -1. Take p-value to be 1 here to overestimate.
        p_vals[k_vals == -1] = 1
        
        return p_vals
        
    def predict(self, eps, test_points=None, scores=None):
        """Return True for each point which is like inputs, False otherwise.
        eps dictates the probability of falsely declaring a point as disimilar
        when it was indeed drawn from the input distribution."""
        # Compute cutoff
        cutoff, success = self.compute_cutoff(eps)
        
        # Score each of the test points
        # Made it so can instead pass in scores to avoid recomputing
        # if not needed
        if scores is None:
            if test_points is None:
                raise ValueError('Either pass in test_points or scores')
            scores = self.compute_scores(test_points)[0]

        # If score <= cutoff then it is considered "like" the input points
        # so return True. Otherwise, return False
        predictions = (scores <= cutoff)
                
        # predictions_temp will match predictions except for rare mistakes due
        # to rounding error when eps is precisely at one of the quantiles.
        # When p_val < eps we reject i.e. False, else accept i.e. True
        # p_vals = self.predict_p(scores=scores)
        # predictions_temp = (p_vals >= eps)
        
        return predictions, cutoff
