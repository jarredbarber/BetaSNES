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
        self.a = np.ones(Nparams,dtype=dtype)
        self.b = np.ones(Nparams,dtype=dtype)
        
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

    def tell(self,X,fit):
        '''Updates the model parameters with the function values.
        '''
        fit = self.utility(fit) # Map to utility values
        # Compute information differentials
        p0a  = pg(0,self.a)
        p0b  = pg(0,self.b)
        p0ab = pg(0,self.a+self.b)

        p1a  = pg(1,self.a)
        p1b  = pg(1,self.b)
        p1ab = pg(1,self.a+self.b)
       
        N = len(X)
        dA = p0ab - p0a + sum([f*np.log(x) for x,f in zip(X,fit)])/N
        dB = p0ab - p0b + sum([f*np.log(1-x) for x,f in zip(X,fit)])/N

        # Compute the Riemannian metric and raise the derivatives
        gdet = p1a*p1b - p1ab*(p1a+p1b)
        gA = ((p1b - p1ab)*dA + p1ab*dB)/(self.a*gdet)
        gB = ((p1a - p1ab)*dB + p1ab*dA)/(self.b*gdet)
        
        # Update parameters w/ exponential map
        self.a *= np.exp(self.step*gA)
        self.b *= np.exp(self.step*gB)

    def stop(self):
        '''Predicate determining if we have hit our convergence criteria'''
        return False # TODO

    def mode(self):
        '''Mode of current distribution'''
        return (self.a-1.0)/(self.a+self.b-2.0)

    def mean(self):
        '''Mean of current distribution'''
        return self.a/(self.a+self.b)

    def var(self):
        '''Variance of current distribution (component wise)'''
        ab = self.a+self.b
        return (self.a*self.b)/( ab*ab*(ab+1) )

    def std(self):
        '''Standard deviation (component wise)'''
        return np.sqrt(self.var)
