import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        EPSILON = EPSILON * EPSILON_DECAY
        while (not done):
            
            cur_state = obs
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample() # currently only performs a random action.
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)
            
            obs,reward,done,info = env.step(action)
            episode_reward += reward # update episode reward

            max_value_of_obs = max([Q_table[(obs,i)] for i in range(env.action_space.n)])
            Q_table[(cur_state, action)] = (1-LEARNING_RATE)*Q_table[(cur_state, action)] + LEARNING_RATE*(reward + DISCOUNT_FACTOR*max_value_of_obs)
        
        
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################