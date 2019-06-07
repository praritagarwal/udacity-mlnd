# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:34:33 2019

@author: Prarit Agarwal
"""
import numpy as np
from collections import deque, namedtuple
from random import sample 

class Replay_Buffer():
    """ replay buffer for storing (s,a,r,s') """
    
    def __init__(self, buffer_size):
        self.memory=deque(maxlen=buffer_size)
        
    def append(self, experience):
        '''append experience to the memory'''
        self.memory.append(experience)
        
    def sample(self, sample_size):
        ''' to withdraw a sample of size sample_size from the replay buffer'''
        return sample(list(self.memory), sample_size)


################## Udacity's Replay Buffer ################################################
        
class UdacityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)    