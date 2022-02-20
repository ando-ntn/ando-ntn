# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:22:01 2021
Project SECO
@author: Xandrous
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import time


"""
replay experience for memory sampling
"""
class DqnReplayBuffer:
    def __init__(self) : 
        self.experiences = deque(maxlen = 1000000)
        #print("replay init done")
        
    
    def record (self, state, next_state, action, reward, done) : 
        self.experiences.append((state, next_state, action, reward, done))
        #print("replay record done")
    
    def sample_batch (self, batch_size) :
        
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for record in sampled_batch:
            state_batch.append(record[0])
            next_state_batch.append(record[1])
            action_batch.append(record[2])
            reward_batch.append(record[3])
            done_batch.append(record[4])
        #print ("replay sample batch done")
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)
        
    
    

"""
Deep Q Network as the brain for the system
"""
class DqnAgent : 
    def __init__ (self, state_space, action_space, gamma, lr, epsilon ) : 
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.model_location = './model'
        self.checkpoint_location = './checkpoints'
        self.mode = "train"
        
        self.q_net = self.build_dqn_model()
        self.target_q_net = self.build_dqn_model()
        
        
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              net=self.q_net)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_location, max_to_keep=10)
        if self.mode == 'train':
            self.load_checkpoint()
            self.update_target_network()
        if self.mode == 'test':
            self.load_model()
            
        #print ("DQN init done")
        
    def build_dqn_model(self) : 
        q_net = keras.Sequential()
        q_net.add(keras.layers.Dense(20, input_dim = self.state_space, activation = 'relu'))
        q_net.add(keras.layers.Dense(10, activation = 'relu'))
        q_net.add(keras.layers.Dense(self.action_space, activation = 'softmax'))
        q_net.compile(optimizer = tf.optimizers.Adam(learning_rate = self.lr), loss ='mse')
        #print ("DQN build model done")
        return q_net
    
    def save_model(self, reward):
        """
        Saves model to file system
        :return: None
        """
        tf.keras.models.save_model(self.q_net, self.model_location)
        file_reward = open('reward.txt', 'w')
        file_reward.write(np.str(reward))
        file_reward.close()
        #print ("DQN save model done")

    def load_model(self):
        """
        Loads previously saved model
        :return: None
        """
        self.q_net = tf.keras.models.load_model(self.model_location)
        #print ("DQN load model done")
        
    def save_checkpoint(self):
        """
        Saves training checkpoint
        :return: None
        """
        self.checkpoint_manager.save()
        #print ("DQN save ckpt done")

    def load_checkpoint(self):
        """
        Loads training checkpoint into the underlying model
        :return: None
        """
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        #print ("DQN load ckpt done")
    
    def update_target_network(self):
        """
        Updates the target Q network with the parameters
        from the currently trained Q network.
        :return: None
        """
        #print('DQN update network done')
        self.target_q_net.set_weights(self.q_net.get_weights())
        
    def train(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, batch_size):
        
        """
        Train the model on a batch
        :return: loss history
        
        input state batch into q_net to get current_q 
        target_q values will be copied from current_q
        input next state batch and we will get the next_q
        we will take max values of next_q (our policy)
        """
        action_batch = action_batch.astype('int16')
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis = 1)    
        """
        in here we will do assigning values to target_q based on the iteration
        and each of previous action taken. 
        
        target_q[current_batch][action_batch[current_batch]] 
        means that the Q values in specific action taken previously 
        target_q[0][1] means q value when taking action 1 in batch 0
        
        done_batch consist of values either 1 or 0
        
        if 1, we will set target_q on specific action equal to the reward
        
        if 0, we will add reward and discounted value from max next_q 
        (normal Q values calculation)
        """
        for current_batch in range(batch_size) : 
            "to check whether current batch already done or not"
            if done_batch[current_batch] : 
                target_q[current_batch][action_batch[current_batch]] = reward_batch[current_batch]
            
            else : 
                target_q[current_batch][action_batch[current_batch]] = reward_batch[current_batch] + self.gamma * max_next_q[current_batch]
        
        "training the model and save it in history"
        history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        loss = history.history['loss']
        #print ("DQN Train done")
        return loss, target_q
    
    def random_policy(self, state):
        """
        Outputs a random action
        :param state: current state
        :return: action
        """
        #print ("DQN random policy done")
        return np.random.randint(0, self.action_space)

    def collect_policy(self, state):
        """
        The policy for collecting data points which can contain 
        some randomness to encourage exploration.
        :return: action
        """
        if np.random.random() < self.epsilon:
            return self.random_policy(state=state)
        #print ("DQN collect policy done")
        return self.policy(state=state)
    
    def policy(self, state):
        """
        Outputs a action based on model
        :param state: current state
        :return: action
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        #print ("DQN policy done")
        return np.int(optimal_action)
    
    def reward_policy(self, total_data) : 
        """
        

        Parameters
        ----------
        feedback_table : user's input 
        if "C" Cold -> -1
        if "N" Normal -> +1
        if "H" Hot -> -1.5
        
        need to define more as in most model RL, rewards are made out of 
        cost function and constraint. in HVAC model-free RL, cost -> Energy Consumption and 
        Constraint -> Comfort Level (temperature and humidty range) need to be combined
        to create reward function as [r = rT + lambda * rp] 
        
        where : rT = cost, lambda = controls tradeoff between cost and constraints
        rp = constraint
        
        Source : https://doi.org/10.1145/3360322.3360861
        
        Returns
        -------
        Reward value

        """
        total_data = np.array(total_data).reshape(1,-1)
        
        feedback_table = total_data[:,-1] 
        watt = total_data[:, 0]
        watt_mean = np.mean(watt)
        reward = []
        
        if watt <= watt_mean : 
            if feedback_table == -1 : 
                feedback_table = -1
            elif feedback_table == 0 : 
                feedback_table = 2
            elif feedback_table == 1 : 
                feedback_table = -1.5
            reward.append(feedback_table) 
        else :
            if feedback_table == -1 : 
                feedback_table = -1
            elif feedback_table == 0 : 
                feedback_table = 2
            elif feedback_table == 1 : 
                feedback_table = -2
            reward.append(feedback_table.map({-1 : -10, 0 : 1, 1:-2}))
                
        return np.float32(reward)
        

"""
for gathering and processing(save laod) data from interface to system
"""

class Processor :
    def __init__ (self) : 
        self.data = 0
        
    
    def preprocess (self, data_file) :   
        
        "for preprocess, don't use pandas. better with tensorflow"
        df = pd.read_csv(data_file)
        data_total = df.copy()
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        
        
        #data_total_day1 = data_total[data_total['Day'] == '5-8-2021']
        #data_total_day2 = data_total[data_total['Day'] == '5/10/2021']
        data_total_day3 = data_total[data_total['Day'] == '5/12/2021']
        data_total_day4 = data_total[data_total['Day'] == '5/14/2021']
        data_total_day5 = data_total[data_total['Day'] == '5/20/2021']
        data_total_day6 = data_total[data_total['Day'] == '5/28/2021']
        data_total_day7 = data_total[data_total['Day'] == '5/29/2021']

        data_total_ready = data_total_day3
        data_total_ready = data_total_ready.append([data_total_day4,data_total_day5,data_total_day6,data_total_day7]) 
        data_total_ready
        
        
        data = data_total_ready
        data = data.drop(['Tot kWH', 'Duration(min)', 'Sensor Placing'], axis = 1)
        data = data.dropna()
        
        for col in ['Temperature', 'RH', 'Heat Index', 'Watt Avg']:
            data[col] = data[col].astype('int64')
            
        x = data[['Heat Index', 'Fan speed']]
        y = data.iloc[:,-1].reset_index().drop(['index'], axis =1 )
        #time = data.iloc[:,0]
        
        
        x = pd.get_dummies(x, drop_first = True)
        columns_values = x.columns.values
        columns_scaled = ['Heat Index']
        
        scaled_x = pd.DataFrame(sc_x.fit_transform(np.reshape([x['Heat Index']], (-1,1))), columns = columns_scaled)
        unscaled_x = x.loc[:,~x.columns.isin(columns_scaled)]
        x = pd.DataFrame(np.hstack([scaled_x,unscaled_x]))
        x.columns = columns_values
        
        y = sc_y.fit_transform(y)
        
        state_data = np.hstack((x.iloc[:, :1], y))
        state_space = len(state_data.shape)
        
        
        "ini untuk nambahin di data table aja sementara"
        reward_avail = [-1, 0, 1]
        reward_index = np.random.randint(3, size = len(state_data))
        reward = []
        for i in range (len(state_data)) :
            reward.append(reward_avail[reward_index[i]])
        reward = np.array(reward).reshape(-1,1)
        #action = np.random.randint(5, size = len(state_data)).reshape(-1,1)
        done = np.zeros((len(state_data),1))
        
        #whole_data = np.concatenate([state_data, reward, action, done], axis = 1)
        whole_data = np.concatenate((state_data, reward, done), axis = 1)
        
        return whole_data, state_space
    
    
    def TrainMode (self, total_data, save_model, load_model, show_loss) : 
        "for training the model after passing certain number of episode"
        if load_model == True : 
            agent.load_model()
            
        reward_total = []
        experience = []
        for i in range (0,len(total_data)-1) : 
            state = total_data[i,:-2]
            next_state = total_data[i+1,:-2]
    
            """
            reward from user's feedback (either 1, 0 or -1)
            action from action taken previously ()
            """
            
            reward = agent.reward_policy(total_data[i,1:3])
            #action = total_data[i,-2:-1].astype(int)
            action = agent.collect_policy(next_state)
            
            
            done = total_data[i,-1:].astype(int)
            replay.record(state, next_state, np.int(action), reward, done)
            print ("eps : %s , Reward : %0.02f" %(i, reward))
            reward_total.append(reward)
            experience.append((replay.experiences[-1][0][0], replay.experiences[-1][0][1],replay.experiences[-1][1][0], replay.experiences[-1][1][1]
                                  ,replay.experiences[-1][2], replay.experiences[-1][3][0]))      
                    
            
            
            "to make sure training run on batch or else it won't train"
            if len(replay.experiences) > batch_size : 
                state_batch, next_state_batch, reward_batch, action_batch, done_batch = replay.sample_batch(batch_size)
                loss, target_q = agent.train(state_batch = state_batch, next_state_batch = next_state_batch
                           , action_batch = action_batch, reward_batch = reward_batch
                           , done_batch = done_batch, batch_size = batch_size)
            "save checkpoint every 50 iteration and update target_q_network"
            if i % 50 == 0 : 
                agent.save_checkpoint()
                agent.update_target_network()
                "decay rate"
                agent.epsilon *= 0.8
                
                
        reward = np.sum(reward_total)       
        last_reward_file = open('reward.txt', 'r')
        last_reward = np.float(last_reward_file.read())
        last_reward_file.close()
        pd.DataFrame(np.array(experience), columns = ['Current_state_heat', 'Current_state_watt', 'Next_State_heat', 'Next_State_watt', 'Action', 'Reward']).to_csv("Train_Memory.csv")
        
        if save_model == True and reward > last_reward : 
            agent.save_model(reward)
            
        if show_loss == True : 
            print("loss : ", np.round(loss,5))
            print("reward : ", reward)
            print('Target_Q : ', np.shape(target_q))
            print("--- %s seconds ---" % (time.time() - start_time)) 
            
    
    def DeployMode (self,new_data) : 
        "for daily operation mode"
        agent.load_model()
        agent.epsilon = 0
        deploy_mode_done = False
        current_state = np.array(new_data[0:2])
        current_reward = agent.reward_policy(new_data[1:-1])
        current_action = agent.collect_policy(current_state)
        current_done = 0
        counter = 1
        experience = []
        while not deploy_mode_done :
        #for i in range (0,100) :
            """
            if current_state is None and current_reward is None : 
                
                #current_state = new_data[:,:-2]
                "delay 1 second untuk nunggu state selanjutnya dann langsung"
                "ambil aja dari value nya jadi ga perlu nunggu submit ke database"

                "reward juga akan ada setelah state muncul bersama user feedback"
                #current_reward = agent.reward_policy(new_data[:,1:-1])
                counter += 1
                """
            
            
            "ini cuma buat test, nanti gati jadi data acquisition"
            state_Heat = np.random.uniform(-1,1)
            state_Y = np.random.uniform(-1,1)
            """setelah dia ngelakuin current action, baru dapet reward.
            reward yang dikirim ketika T+1 ini akan jadi reward(T)"""
            reward_avail = [-1,0,1]
            reward_index = np.random.randint(3)
            reward = reward_avail[reward_index]
            "test doang ya sampe sini."
                
            "State (T+1)"
            #new_state = new_data[:,:-2]
            new_state =[ state_Heat, state_Y]
            new_state = np.array(new_state)
            
            current_reward = agent.reward_policy(reward)
            
            replay.record(current_state, new_state, np.int(current_action), current_reward, current_done)
                
            new_action = agent.collect_policy(new_state)
                
            "re-assigning current state, action and reward from new data"
            current_state = new_state
            current_action = new_action
                
            print("state now : ", current_state)
            print("action taken : ", current_action)
            print("reward : ", current_reward)
            print('--------------------------------------------')
                
            "nanti disini musti ditambah untuk save data tadi ke dalam local storage"
            "untuk nanti kita bisa save ke database periodically at the end of the day for training"
            counter += 1
            time.sleep(0.1)
                
            "ini untuk save  data nya taruh disini dlu. save setiap 50 step"
            experience.append((replay.experiences[-1][0][0], replay.experiences[-1][0][1],replay.experiences[-1][1][0], replay.experiences[-1][1][1]
                                  ,replay.experiences[-1][2], replay.experiences[-1][3][0]))   
                    
            if counter % 50 == 0 : 
                pd.DataFrame(np.array(experience), columns = ['Current_state_heat', 'Current_state_watt', 'Next_State_heat', 'Next_State_watt', 'Action', 'Reward']).to_csv("Deploy_Memory.csv")
        print("--- %s seconds ---" % (time.time() - start_time))    


start_time = time.time()
        
"config"
batch_size = 40
gamma = 0.95
lr= 0.001    
"action_space in this case we just mention either increase or decrease fan speed or temp (do nothing, up temp, down temp, up fan, or down fan)"
action_space = 5
epsilon = 0.05
save_model = True
load_model = True
show_loss = True

print('Welcome to DQN Algorithm for SECO')
mode = input('Config Mode (Deploy/Train): ')

processor = Processor()
total_data, state_space= processor.preprocess('Energy Data - Data kamar ando kerja.csv')
""""

Total_data nanti bisa dipisah jadi data dari database buat trainperiod dan
trainauto yang langsung di device itu dia ambil directly dari data acquisitionnya

"""
agent = DqnAgent(action_space = action_space , state_space = state_space, 
                 gamma = gamma, lr = lr, epsilon = epsilon)
replay = DqnReplayBuffer()


"Ini buat testing doang, nanti jangan lupa di hapus"
new_data = [1.2, 0.9, 1, None]


if mode == "Train" : 
    processor.TrainMode(total_data, save_model = save_model, 
                      load_model = load_model, show_loss = show_loss)
elif mode == "Deploy" : 
    processor.DeployMode(new_data = new_data)
    

    
action_taken = []
for a in range (len(replay.experiences)) : 
    action_taken.append(replay.experiences[a][2])
    
print(action_taken)    
