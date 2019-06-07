# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:46:36 2019

Based on the implementation in Udacity's code for RL-Quadcopter 2 

@author: Prarit Agarwal
"""

import numpy as np
from copy import copy


class OU_noise():
    
    def __init__(self, mu, sigma, theta, action_size):
        self.action_size=action_size
        self.mu=mu*np.ones(action_size)
        self.sigma=sigma
        self.theta=theta
        self.reset()
        
    def reset(self, sigma=None, theta=None):
        self.xt=copy(self.mu) # need to use the copy function because direct assignment creates a binding whihc makes self.mu also change when self.xt is changed to the next step
        if not sigma==None: 
           self.sigma=sigma
        if not theta==None:   
           self.theta=theta
    
    def ou_noise(self):
        
        dx=self.theta*(self.mu-self.xt) + self.sigma*np.random.randn(self.action_size)
        self.xt+=dx
        return self.xt
        
        
################################ Udacity's OUNoise ########################
        
class UdacityOUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state    