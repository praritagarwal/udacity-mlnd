# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:13:52 2019

@author: Prarit Agarwal
"""

from keras.models import Model
from keras.layers import Dense, Concatenate, Input, BatchNormalization, Add, Activation
from keras import optimizers
import keras.backend as K

class Critic():
    """ this class will create a critic network """
    
    def __init__(self, state_size, action_size):
        self.state_size=state_size
        self.action_size=action_size
        
		# call the critic_DNN to build the model
        self.critic_DNN()

    def critic_DNN(self):
        """function to create the critic's dense neural network"""
        state_input=Input(shape=(self.state_size,))
        action_input=Input(shape=(self.action_size,), name='actions')
        hidden_state1=Dense(units=32, activation='relu')(state_input)
        norm_hidden_state1=BatchNormalization()(hidden_state1)
        hidden_state2=Dense(units=64, activation='relu')(norm_hidden_state1)
        norm_hidden_state2=BatchNormalization()(hidden_state2)
        hidden_action1=Dense(units=32, activation='relu')(action_input)
        norm_hidden_action1=BatchNormalization()(hidden_action1)
        hidden_action2=Dense(units=64, activation='relu')(norm_hidden_action1)
        norm_hidden_action2=BatchNormalization()(hidden_action2) 
        concat=Concatenate()([norm_hidden_state2, norm_hidden_action2])
        hidden1=Dense(units=32, activation='relu')(concat)
        norm_hidden1=BatchNormalization()(hidden1)
        hidden2=Dense(units=64, activation='relu')(norm_hidden1)
        norm_hidden2=BatchNormalization()(hidden2)
        hidden3=Dense(units=32, activation='relu')(norm_hidden2)
        norm_hidden3=BatchNormalization()(hidden3)
        #hidden4=Dense(units=8, activation='relu')(norm_hidden3)
        #norm_hidden4=BatchNormalization()(hidden4)
        output_q=Dense(units=1)(norm_hidden3)
        self.model = Model(inputs=[state_input, action_input], outputs=output_q)
        
		# Note that the layers that are used to define the model are not class variables. Thus they cannot be used from outside of the function. 
		# Therefore we have to define the function that returns the action_gradients from within critic_DNN itself
		# We will use K.gradients to compute the action gradients
		# And then define a function inside critic_DNN to return the action_gradients whenever required
		# We will do this through K.function
		# In this way the critic_DNN also acts like a factory function that creates a function to return action_gradients
		# This way of obtaining action gradients is from udacity's Critic class

        action_gradients = K.gradients(output_q, action_input)
		
        self.get_action_gradients=K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients )
        # here the '*' in *self.model.input is used in the sense of *args
        # needed to include this because there are more than 1 model inputs (the states and the actions) here
        
################################## Udacity's Critic ################################        
    
class UdacityCritic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = Input(shape=(self.state_size,), name='states')
        actions = Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = Dense(units=32, activation='relu')(states)
        net_states = Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = Dense(units=32, activation='relu')(actions)
        net_actions = Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = Add()([net_states, net_actions])
        net = Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

        
        
