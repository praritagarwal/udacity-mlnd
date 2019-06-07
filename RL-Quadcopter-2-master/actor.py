# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:13:52 2019

@author: Prarit Agarwal
"""

from keras.models import Model
from keras.layers import Dense, Lambda, Input, BatchNormalization
from keras import optimizers
import keras.backend as K

class Actor():
    """ this class will create a critic network """
    
    def __init__(self, state_size, action_size, action_low, action_high, learning_rate=0.001):
        self.state_size=state_size
        self.action_size=action_size
        self.action_low=action_low
        self.action_high=action_high
        self.learning_rate=learning_rate
        
        self.actor_DNN()
        
        
        
    def actor_DNN(self):
        """ function to create the critic's dense neural network """
        state_input=Input(shape=(self.state_size,))
        hidden1=Dense(units=32, activation='relu')(state_input)
        norm_hidden1=BatchNormalization()(hidden1)
        hidden2=Dense(units=64, activation='relu')(norm_hidden1)
        norm_hidden2=BatchNormalization()(hidden2)
        hidden3=Dense(units=32, activation='relu')(norm_hidden2)
        norm_hidden3=BatchNormalization()(hidden3)
        #hidden4=Dense(units=8, activation='relu')(norm_hidden3)
        #norm_hidden4=BatchNormalization()(hidden4)
        action=Dense(units=self.action_size, activation='sigmoid')(norm_hidden3)
        rescaled_action=Lambda(lambda x: x*(self.action_high-self.action_low)+self.action_low)(action)
        
        self.model =Model(inputs=state_input, outputs=rescaled_action)
        
        # actor's loss function
        action_gradients=Input(shape=(self.action_size,))
        loss=K.mean(-action_gradients*rescaled_action)
        
        #updates
        #optimizer=optimizers.Adam(lr=self.learning_rate, amsgrad=True)
        optimizer=optimizers.Nadam()
        update_op=optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.training=K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=update_op)
        
################################# Udacity's Actor ##########################################
        
class UdacityActor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = Dense(units=32, activation='relu')(states)
        net = Dense(units=64, activation='relu')(net)
        net = Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)        


        
        
