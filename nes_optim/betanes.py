from __future__ import absolute_import
import numpy as np
from scipy.special import polygamma as pg
import .util_fns

class BetaNES:
    dtype=np.float32
    def __init__(self,Nparams):
        '''Constructor.
        Initializes the optimizer with a maximum entropy distribution.
        '''
        # Distribution parameters
        self.theta = np.ones(Nparams,2,dtype=dtype)
        
        # Optimization parameters
        self.popsize = int(4+3*np.log(Nparams)) # Population size (from CMA-ES)
        self.utility = util_fns.identity

        # Bookkeeping stuff
        self.fmean = None  # Function mean estimate
        self.step  = 1.0   # Step size

    def ask(self):
        '''Proposes a set of self.popsize solutions from the current distribution.
        '''
        return [np.random.beta(self.theta[:,0],self.theta[:,1]) for ii in range(self.popsize)]

    def tell(self,X,f):
        '''Updates the model parameters with the function values.
        '''
        f = self.utility(f) # Map to utility values
        # Compute information gradients
        #TODO
        # Compute the Riemannian metric
        #TODO
        # Update parameters w/ exponential map
        #TODO

    def stop(self):
        '''Predicate determining if we have hit our convergence criteria'''
        return False # TODO

    def mode(self):
        '''Mode of current distribution'''
        a = self.params[:,0]
        b = self.params[:,1]
        return (a-1.0)/(a+b-2.0)

    def mean(self):
        '''Mean of current distribution'''
        a = self.params[:,0]
        b = self.params[:,1]
        return (a)/(a+b)

    def var(self):
        '''Variance of current distribution (component wise)'''
        a = self.params[:,0]
        b = self.params[:,1]
        return (a*b)/( (a+b)*(a+b)*(a+b+1) )

    def std(self):
        '''Standard deviation (component wise)'''
        return np.sqrt(self.var)
