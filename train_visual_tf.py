import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Parámetros del entorno
ENV_NAME = "LunarLander-v2"
env = gym.make(ENV_NAME, render_mode="human")

# Parámetros del agente DQN
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n
LEARNING_RATE = 0.001
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MIN_MEMORY_SIZE = 1000

def build_model():
    model = Sequential()
    model.add(tf.keras.Input(shape=(STATE_SIZE,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()
        self.update_target_model()
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        state = np.reshape(state, [1, STATE_SIZE])
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        minibatch = self.memory.sample(batch_size)
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * np.amax(q_next[i])
            q_values[i][actions[i]] = target
        self.model.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

def train_dqn(episodes, render_every_n=10):
    agent = DQNAgent()
    batch_size = BATCH_SIZE
    
    for e in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, STATE_SIZE])
        total_reward = 0
        done = False

        render = True if (e + 1) % (render_every_n + 1) == 0 else False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, STATE_SIZE])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if render:
                env.render()
                time.sleep(0.005)

        agent.replay(batch_size)

        if e % 20 == 0:
            agent.update_target_model()

        print(f"Episodio: {e}/{episodes}, Recompensa Total: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    env.close()

train_dqn(1000, render_every_n=100)
