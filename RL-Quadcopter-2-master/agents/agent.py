import numpy as np


from actor import Actor, UdacityActor
from critic import Critic, UdacityCritic
from replay import Replay_Buffer, UdacityReplayBuffer
from Ornstein_Uhlenbeck_noise import OU_noise, UdacityOUNoise # to add a little noise to the action for purpose of exploration
from keras import optimizers
from collections import deque


class Agent():
    """ DDPG based learning agent. My own implementation """
    
    def __init__(self, env, buffer_size=100000, gamma=0.99, tau = 0.01, mu = 0.0, sigma = 0.5, theta = 0.15, sample_size=64, learning_rate=0.001, learning_mode=1 ):
        #original sigma = 0.2
        #next, I tried sigma =0.5 with some successs
        self.env=env
        self.action_low, self.action_high=self.env.action_bounds()
        self.gamma=gamma
        self.tau=tau
        self.learning_rate=learning_rate
        self.learning_mode=learning_mode # to control when the agent is training 
        
        # check if the agent is stuck in a local minima by checking the variance of the scores
        self.last_n_scores=deque(maxlen=250)
        
        # creating an instance of OU_noise
        self.sigma=sigma
        self.theta=theta
        self.ou_process = OU_noise(mu, self.sigma, self.theta, self.env.action_size)
        self.exploration_rate=1 
        
        # create a local and a target actor/crtic network
        self.local_actor=Actor(state_size=self.env.state_size, action_size=self.env.action_size,
                               action_low=self.action_low, action_high=self.action_high, learning_rate=self.learning_rate)
        self.local_critic=Critic(state_size=self.env.state_size, action_size=self.env.action_size)
        
        self.target_actor=Actor(state_size=self.env.state_size, action_size=self.env.action_size,
                               action_low=self.action_low, action_high=self.action_high)
        self.target_critic=Critic(state_size=self.env.state_size, action_size=self.env.action_size)
        
        #Initially, we will copy the local weights of the actor and critic networks to those of the target actor/critic networks
        self.target_actor.model.set_weights(self.local_actor.model.get_weights())
        self.target_critic.model.set_weights(self.local_critic.model.get_weights())
        
        
        # compiling the local critic
        #optimizer=optimizers.Adam(lr=self.learning_rate, amsgrad=True)
        optimizer=optimizers.Nadam()
        self.local_critic.model.compile(optimizer=optimizer, loss='mse')
        
        # create a memory replay to store (state, action, reward, next_state)
        self.replay_buffer=Replay_Buffer(buffer_size=buffer_size)
        self.sample=None # sample of experiences from the replay_buffer
        #self.min_experiences = min_experiences  # minimum number of experiences required in the replay buffer to start training
        self.sample_size=sample_size            # the sample size to be drawn from the replay buffer for training
        
        # the action corresponding to the next_state as generated by the target actor.
        # y_target is the target q-value of the current state
        self.target_next_actions=None
        self.y_target=None
        
        self.steps=1 # to count the number of steps taken by the agent in a given episode
        self.score=None # to keep track of the score (total discounted rewards for the begining) in each episode
        self.best_score=-np.inf # to store the score of the best episode
        self.noise_scale=None
        self.average_pos=None # keep track of the average position of the quad over the episode
        
    
    def reset(self):
        """ function to reset the environment.
            Update self.best_score if self.score is better. 
            Reset self.score to zero and self.steps to 1. 
            We will also reset the ou_process   """
        state = np.array(self.env.reset())
        
        if not(self.score==None):
           if self.score>self.best_score:
              self.best_score=self.score
        
        # increase sigma if the scores have plateaued 
        #if len(self.last_n_scores)>=250 and np.var(self.last_n_scores)<=50.0:
        #    self.sigma=np.min([1, 1.002*self.sigma])
        #    self.exploration_rate=1
        #    print("###############scores plateaued; increasing sigma and resetting explorating rate!!!#############")
        
        
        self.steps=1
        self.score=0
        self.average_pos=np.array([0., 0., 0.])
        self.ou_process.reset(sigma=self.sigma, theta=self.theta)
        self.exploration_rate*=0.999
        return state
        
    def action(self, state):
        state=state.reshape(1,-1)
        # predict the action values using the local_actor
        # add OU noise
        # clip the action values to lie with action_low and action_high
        noise=self.learning_mode*self.exploration_rate*self.ou_process.ou_noise()
        self.noise_scale=np.max(noise)
        return np.clip(self.local_actor.model.predict(x=state)[0]+noise
                        + 0.0001*np.ones(self.env.action_size), self.action_low + 0.0001, self.action_high)
    
    def step(self, state, action):
        next_state, reward, done, success_rate, success = self.env.step(action)
        experience=[state, action, reward, next_state, done]
        self.replay_buffer.append(experience)
        
        # add rewards to self.score and increase self.steps by 1
        self.score+=reward
        self.average_pos+=next_state.reshape(3,-1).mean(axis=0)[:3]
        self.steps+=1
        
        # if done, then average the score over the number of steps
        if done:
            self.score=self.score/self.steps
            self.last_n_scores.append(self.score)
            self.average_pos=self.average_pos/self.steps
            if np.array(success_rate).sum()>=8:
                self.learning_mode=0
        
        # start training if and when there are atleast  min_experiences in the replay_buffer
        # for training, draw a sample of sample_size from the replay buffer
        #min_experiences=self.min_experiences
        #sample_size=self.sample_size
        if (len(self.replay_buffer.memory)>=self.sample_size and self.learning_mode==1):
            self.sample=self.replay_buffer.sample(self.sample_size)
            self.sample_states=np.array([exp[0] for exp in self.sample]) # initial states in the samples
            self.sample_actions=np.array([exp[1] for exp in self.sample]) # the actions taken in the samples
            self.sample_next_states=np.array([exp[3] for exp in self.sample]) # next_states in the samples
            self.rewards=np.array([exp[2] for exp in self.sample]).reshape(-1,1) # rewards in the sample. Make sure to reshape
            self.dones=np.array([exp[4] for exp in self.sample]).reshape(-1,1)
            
            self.update_local_critic()
            self.update_target_critic()
            self.update_local_actor()
            self.update_target_actor()
        
        return next_state, reward, done, success
    
    def update_local_critic(self):
        self.target_next_actions=self.target_actor.model.predict_on_batch(x=self.sample_next_states) # actions in next_state as predicted by the target_actor
        target_next_qs=self.target_critic.model.predict_on_batch([self.sample_next_states, self.target_next_actions]) # q_value in the next state as predicted by the target_critic 
        self.y_target=self.rewards+self.gamma*target_next_qs
        self.local_critic.model.train_on_batch(x= [self.sample_states, self.sample_actions], y=self.y_target)
        
    def update_target_critic(self):
        local_critic_weights=np.array(self.local_critic.model.get_weights())
        target_critic_weights=np.array(self.target_critic.model.get_weights())
        target_critic_weights+=self.tau*(local_critic_weights-target_critic_weights)
        self.target_critic.model.set_weights(target_critic_weights)
        
    def update_local_actor(self):
        self.action_gradients=np.array(self.local_critic.get_action_gradients([self.sample_states, self.sample_actions, 0])).reshape(-1, self.env.action_size)
        self.local_actor.training([self.sample_states, self.action_gradients, 1])
        
    def update_target_actor(self):
        local_actor_weights=np.array(self.local_actor.model.get_weights())
        target_actor_weights=np.array(self.target_actor.model.get_weights())
        target_actor_weights+=self.tau*(local_actor_weights-target_actor_weights)
        self.target_actor.model.set_weights(target_actor_weights)


####################################Udacity's Agent ###########################################################        
    
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, tau=0.01, gamma=0.99, buffer_size=100000, sample_size=64, mu=0.0, theta=0.15, sigma=0.2):
        self.task = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.action_low = env.action_low
        self.action_high = env.action_high

        # Actor (Policy) Model
        self.actor_local = UdacityActor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = UdacityActor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = UdacityCritic(self.state_size, self.action_size)
        self.critic_target = UdacityCritic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = mu
        self.exploration_theta = theta
        self.exploration_sigma = sigma
        self.noise = UdacityOUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = sample_size
        self.memory = UdacityReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)    



    
    
    