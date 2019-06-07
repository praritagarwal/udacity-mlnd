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
        self.start_pos=init_pose
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        # distance between initial position and target position 
        self.total_distance= (((self.target_pos-self.init_pos)**2).sum())**(0.5)
          
        
           
        
    def get_delta_distance_reward(self, previous_pos, new_pos):
        """Uses the evolution of pose with respect to the target to return a reward.
        1) positive reward if the distance to the target has decreased
        2) negative reward if the distance to the target has increased
        """
        #cosine = np.dot(self.target_pos-previous_pos, new_pos-previous_pos)/((np.dot(self.target_pos-previous_pos,self.target_pos-previous_pos) * np.dot(new_pos-previous_pos,new_pos-previous_pos))**(0.5)+0.0001)
        dist_new=np.dot(new_pos-self.target_pos,new_pos-self.target_pos)**(0.5)
        dist_old=np.dot(previous_pos-self.target_pos,previous_pos-self.target_pos)**(0.5)
        if dist_new<=0.1:
            reward=750*self.runtime
            print("\nSUCESS!!!!!")
            self.success=1
                #if cosine>=0:
                #reward=self.runtime
        elif dist_new<=0.5:
            reward=500*self.runtime
            print("$$$$$ Almost there $$$$$")        
        elif dist_new<=1:
            reward=250*self.runtime
            print("$$$$$ Getting there $$$$$")
            #if cosine >= 0.0:
                #reward=4*self.runtime/5
        elif self.hit_the_bounds(new_pos):
            reward=-50*self.runtime/(3)
            #if cosine>=0:
                #reward=1/3
        else:
            reward=np.sign(dist_old-dist_new)
            #reward=cosine/3.0+1/3.0+5/dist_new
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    
    #Alternate rewards scheme 
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
        if dist_new<=0.1:
            reward=750*self.runtime
            print("\nSUCESS!!!!!")
            self.success=1
                #if cosine>=0:
                #reward=self.runtime
        elif dist_new<=0.5:
            reward=500*self.runtime
            print("$$$$$ Almost there $$$$$")        
        elif dist_new<=1:
            reward=250*self.runtime
            print("$$$$$ Getting there $$$$$")
            #if cosine >= 0.0:
                #reward=4*self.runtime/5
        elif self.hit_the_bounds(new_pos):
            reward=-50*self.runtime/(3)
            #if cosine>=0:
                #reward=1/3
        else:
            reward=cosine
            #reward=cosine/3.0+1/3.0+5/dist_new
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
   
    def hit_the_bounds(self, pos):
        hit=False
        margin=0.1
        if np.abs(self.x_lower_bound-pos[0])<=margin :
            hit = True
            print("HIT X_LOWER !!!!")
            return hit
        elif np.abs(self.x_upper_bound-pos[0])<=margin :
            hit = True
            print("HIT X_UPPER !!!!")
            return hit
        elif np.abs(self.y_lower_bound-pos[1])<=margin :
            hit = True
            print("HIT Y_LOWER !!!!")
            return hit
        elif np.abs(self.y_upper_bound-pos[1])<=margin :
            hit = True
            print("HIT Y_UPPER !!!!")
            return hit
        elif np.abs(self.z_lower_bound-pos[2])<=margin :
            hit = True
            print("HIT Z_LOWER !!!!")
            return hit
        elif np.abs(self.z_upper_bound-pos[2])<=margin :
            hit = True
            print("HIT Z_UPPER !!!!")
            return hit
        
    def get_reward(self, previous_pos, new_pos, cos=0):
        if cos==0:
            reward= self.get_delta_distance_reward(previous_pos, new_pos)
        else:
            reward= self.get_cosine_reward(previous_pos, new_pos)
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            previous_pos=self.sim.pose[:3] # pose of the copter before applying the rotor_speeds; We will only use the x,y,z coordinates
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            new_pos=self.sim.pose[:3] # pose of the copter after applying the rotor_speeds; We will only use the x,y,z coordinates
            reward += self.get_reward(previous_pos, new_pos, cos=0) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        if self.success==1:
            done=True
            self.success_rate.append(1)
        elif done:
            self.success_rate.append(0)
        return next_state, reward, done, self.success_rate

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
        
        
        
        
        
        
        
        
        
        
    