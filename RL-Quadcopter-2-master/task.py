import numpy as np
from physics_sim import PhysicsSim
from collections import deque


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.success=0
        self.success_rate=deque(maxlen=10)  # to record agent's success/failure for 10 consecutive episodes 
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime=runtime
        self.x_lower_bound=self.sim.lower_bounds[0]
        self.y_lower_bound=self.sim.lower_bounds[1]
        self.z_lower_bound=self.sim.lower_bounds[2]
        self.x_upper_bound=self.sim.upper_bounds[0]
        self.y_upper_bound=self.sim.upper_bounds[1]
        self.z_upper_bound=self.sim.upper_bounds[2]
        
        #Initial pos
        self.start_pos=init_pose[:3]
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        # distance between initial position and target position 
        self.total_distance= (np.dot(self.target_pos-self.start_pos, self.target_pos-self.start_pos))**(0.5)

        self.target_margin=self.total_distance/50 # this is equal to 2% of the starting distance
        
        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            previous_pos=self.sim.pose[:3] # pose of the copter before applying the rotor_speeds; We will only use the x,y,z coordinates
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            new_pos=self.sim.pose[:3] # pose of the copter after applying the rotor_speeds; We will only use the x,y,z coordinates
            reward += self.get_reward(previous_pos, new_pos, scheme=1) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        if self.success==1:
            done=True
            self.success_rate.append(1)
        elif done:
            self.success_rate.append(0)
        return next_state, reward, done, self.success_rate, self.success

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.success=0
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
    
    def state_size(self):
        '''function to return state_size'''
        return self.state_size
    
    def action_size(self):
        '''function to return action_size'''
        return self.action_size
    
    def action_bounds(self):
        '''function to return the bounds on the action space'''
        return self.action_low, self.action_high
 
    
    def get_reward(self, previous_pos, new_pos, scheme=0):
        if scheme==0:
            reward= self.get_cosine_reward(previous_pos, new_pos)
        elif scheme==1:
            reward= self.get_delta_distance_reward_1(previous_pos, new_pos)
        elif scheme==2:
            reward= self.get_delta_distance_reward_2(previous_pos, new_pos)
        elif scheme==3:
            reward= self.get_delta_distance_reward_3(previous_pos, new_pos)
        elif scheme==4:
            reward= self.get_delta_distance_reward_4(previous_pos, new_pos)    
        return reward
     
    # uncomment the print statements if required    
    def hit_the_bounds(self, pos):
        """
        Function to check if the quad has approached within a distance of 0.1 of the boundaries. 
        """
        hit=False
        margin=0.1
        if np.abs(self.x_lower_bound-pos[0])<=margin :
            hit = True
            #print("HIT X_LOWER !!!!")
            return hit
        elif np.abs(self.x_upper_bound-pos[0])<=margin :
            hit = True
            #print("HIT X_UPPER !!!!")
            return hit
        elif np.abs(self.y_lower_bound-pos[1])<=margin :
            hit = True
            #print("HIT Y_LOWER !!!!")
            return hit
        elif np.abs(self.y_upper_bound-pos[1])<=margin :
            hit = True
            #print("HIT Y_UPPER !!!!")
            return hit
        elif np.abs(self.z_lower_bound-pos[2])<=margin :
            hit = True
            #print("HIT Z_LOWER !!!!")
            return hit
        elif np.abs(self.z_upper_bound-pos[2])<=margin :
            hit = True
            #print("HIT Z_UPPER !!!!")
            return hit
        
        
    
    #Upon testing, I found that the cosine_rewards was not a very good scheme
    #The quad did not even learn to reach a target which was vertically above its starting position
    def get_cosine_reward(self, previous_pos, new_pos):
        """Uses the evolution of pose with respect to the target to return a reward.
           1) Consider the two displacement vectors: target-previous_pose, new_pose-previous_pose 
           2) Positive reward if the new_pose is in the same direction as the target. This implies that the dot_product of the 
              two displacement vectors is positive
           3) negative reward if the dot product between the two displacement vectors is negative, since this will imply that 
              the new pose is in the opposite direction w.r.t to the target 
           4) This suggests that using the cosine of the angle between the two vectors might be a good strategy   
        """
        cosine = np.dot(self.target_pos-previous_pos, new_pos-previous_pos)/((np.dot(self.target_pos-previous_pos,self.target_pos-previous_pos) * np.dot(new_pos-previous_pos,new_pos-previous_pos))**(0.5)+0.0001)
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        #dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_new<=2.5*self.target_margin:
            reward=1000*self.total_distance
            print("\nSUCESS!!!!!")
            self.success=1
                
        elif dist_new<=5*self.target_margin:
            reward=15*self.total_distance*cosine
            print("$$$$$ Almost there $$$$$")
                       
        elif dist_new<=10*self.target_margin:
            reward=7.5*self.total_distance*cosine
            print("$$$$$ Getting there $$$$$")
            
        elif self.hit_the_bounds(new_pos):
            reward=-100*self.total_distance/3
            
        else:
            reward=5*cosine
            
        return reward    
    
    
    #This worked very nicely for a target that was vertically above the starting position. 
    #The quad accomplished the task of reaching a target vertically above it,  after about 250 episodes   
    # uncomment the print statements if required 
    def get_delta_distance_reward_1(self, previous_pos, new_pos):
        """Uses the evolution of pose with respect to the target to return a reward.
        1) positive reward if the distance to the target has decreased
        2) negative reward if the distance to the target has increased
        """
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_new<=2.5*self.target_margin:
            reward=1000*self.runtime
            #print("\nSUCESS!!!!!")
            self.success=1
        elif dist_new<=5*self.target_margin:
            reward=50*self.runtime
            #print("$$$$$ Almost there $$$$$")        
        elif dist_new<=10*self.target_margin:
            reward=25*self.runtime
            #print("$$$$$ Getting there $$$$$")
        elif self.hit_the_bounds(new_pos):
            reward=-500*self.runtime/(3)
        else:
            reward=np.sign(dist_old-dist_new)
        return reward   
    
    # This worked for a target that was positioned vertically above the starting position 
    # The quad learnt to reach the target after about 500 episodes
    def get_delta_distance_reward_2(self, previous_pos, new_pos):
        """Uses the evolution of pose with respect to the target to return a reward.
        1) positive reward if the distance to the target has decreased
        2) negative reward if the distance to the target has increased
        """
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_new<=2.5*self.target_margin:
            reward=1000*self.runtime
            print("\nSUCESS!!!!!")
            self.success=1
        elif dist_new<=5*self.target_margin:
            reward=50*self.runtime
            print("$$$$$ Almost there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
               reward-=50
            else:
                print("%%%% Moving closer :) %%%%")            
        elif dist_new<=10*self.target_margin:
            reward=25*self.runtime
            print("$$$$$ Getting there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%") 
               reward-=25
            else:
                print("%%%% Moving closer :) %%%%")
        elif dist_new<=20*self.target_margin:
            reward=5*self.runtime
            if np.sign(dist_old-dist_new)==-1:
               reward-=15
        elif dist_new<=40*self.target_margin:
            reward=2*self.runtime
            if np.sign(dist_old-dist_new)==-1:
               reward-=5
        elif dist_new<=45*self.target_margin:
            reward=self.runtime
            if np.sign(dist_old-dist_new)==-1:
               reward-=3
        elif self.hit_the_bounds(new_pos):
            print("penalty!!!")
            reward=-500*self.runtime/(3)
        else:
            reward=np.sign(dist_old-dist_new)
        return reward

    # This worked for a target positioned vertically above 
    # Did not work for a more general target
    def get_delta_distance_reward_3(self, previous_pos, new_pos):
        
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_old<=2.5*self.target_margin:
            reward=self.total_distance*100
            print("\nSUCESS!!!!!")
            self.success=1
                #if cosine>=0:
                #reward=self.runtime
        elif self.hit_the_bounds(new_pos):
            reward=-100*self.total_distance
        else:
            reward=-7.5*(dist_old/self.total_distance)/100
 
        if dist_new<=4*self.target_margin:
            print("$$$$$ A little bit more $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
               reward=-self.total_distance*30
            else:
                print("%%%% Moving closer :) %%%%") 
                #reward+=7*(1-dist_old/self.total_distance)/100 
                reward=self.total_distance*50
        if dist_new<=5*self.target_margin:
            print("$$$$$ Almost there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
               reward=-self.total_distance*15
            else:
                print("%%%% Moving closer :) %%%%") 
                #reward+=5*(1-dist_old/self.total_distance)/100
                reward=self.total_distance*25
        elif dist_new<=10*self.target_margin:
            print("$$$$$ Getting there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
               reward-=2.5*(1-dist_old/self.total_distance)/100
            else:
                print("%%%% Moving closer :) %%%%")
                reward+=2.5*(1-dist_old/self.total_distance)/100   
        
        return reward   
       
    #This worked for a target vertcally above, with the noise parameters being: sigma = 0.5, theta = 0.15 
    #Did not work for a more general target position
    def get_delta_distance_reward_4(self, previous_pos, new_pos):
        
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_old<=2.5*self.target_margin:
            reward=self.total_distance*100
            print("\nSUCESS!!!!!")
            self.success=1
        elif self.hit_the_bounds(new_pos):
            reward=-100*self.total_distance
        else:
            reward=-7.5*((dist_old/self.total_distance)**0.2)/100 # The idea of rasing to a power less than 1, came from https://www.youtube.com/watch?v=0R3PnJEisqk
 
        if dist_new<=4*self.target_margin:
            print("$$$$$ A little bit more $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
            else:
                print("%%%% Moving closer :) %%%%")           
        elif dist_new<=5*self.target_margin:
            print("$$$$$ Almost there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
            else:
                print("%%%% Moving closer :) %%%%")                
        elif dist_new<=10*self.target_margin:
            print("$$$$$ Getting there $$$$$")
            if np.sign(dist_old-dist_new)==-1:
               print("%%%%%% Moving away :( %%%%")
            else:
                print("%%%% Moving closer :) %%%%")
       
        return reward   
        
        
    