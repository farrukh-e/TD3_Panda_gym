#TODO:
# 1. Implement the TD3 algorithm
# 2. Integrate the TD3 algorithm with the PandaProject
# 3. Test the TD3 algorithm with the PandaProject
# 4. Add imitation learning to the TD3 algorithm

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Input 
from tensorflow.keras.optimizers import Adam
import os

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, *input_shape))
        self.next_state = np.zeros((max_size, *input_shape))
        self.action = np.zeros((max_size, n_actions))
        self.reward = np.zeros(max_size)
        self.terminal = np.zeros(max_size, dtype=bool)

    # Create a buffer to store the experience tuples
    # ptr is a circular buffer pointer
    def add(self, state, action, reward, next_state, done):
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.terminal[self.ptr] = done


    def sample(self, batch_size):
        # max_mem points to the maximum number of experiences in the buffer
        max_mem = min(self.ptr, self.max_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        next_states = self.next_state[batch]
        dones = self.terminal[batch]
        return states, actions, rewards, next_states, dones
    

class Critic(keras.Model):
    def __init__(self, state_dim, action_dim, name="critic"):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name

        # Q1 architecture
        self.l1 = Dense(256, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(1)

        # Q2 architecture
        self.l4 = Dense(256, activation='relu')
        self.l5 = Dense(256, activation='relu')
        self.l6 = Dense(1)

    # Create the Q1 and Q2 networks
    def call(self, state, action):
        sa = tf.concat([state, action], axis=-1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(sa)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = tf.concat([state, action], axis=-1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        
        return q1
    
class Actor(keras.Model):
    def __init__(self, state_dim, action_dim, max_action, name="actor"):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = Dense(256, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(action_dim, activation='tanh')

    def call(self, state):
        prob = self.l1(state)
        prob = self.l2(prob)
        prob = self.l3(prob)
        return prob * self.max_action
    
class TD3:
    def __init__(
		self,
		state_dim,
		action_dim,
        max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Initialize the actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.set_weights(self.actor.get_weights())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.actor_optimizer = Adam(learning_rate=3e-4)
        self.critic_optimizer = Adam(learning_rate=3e-4)

        # Initialize the training step counter
        self.total_it = 0

    def select_action(self, state):
        state = np.array(state).reshape(1, -1)
        action = self.actor(state).numpy()
        return action.flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        not_dones = 1 - dones

        noise = tf.clip_by_value(
        tf.random.normal(tf.shape(actions), stddev=self.policy_noise),
        -self.noise_clip, self.noise_clip)

        next_actions = tf.clip_by_value(
            self.actor_target(next_states) + noise,
            -self.max_action, self.max_action)

        target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = rewards[:, None] + (1.0 - dones[:, None]) * self.discount * target_Q
        target_Q = tf.stop_gradient(target_Q) 
        
        # ===== CRITIC UPDATE =====
        with tf.GradientTape() as tape:            
            current_Q1, current_Q2 = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(current_Q1 - target_Q)) + \
                        tf.reduce_mean(tf.square(current_Q2 - target_Q))
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # ===== ACTOR UPDATE (DELAYED) =====
        if self.total_it % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                # Compute actor loss
                actor_actions = self.actor(states)
                actor_loss = - \
                    tf.reduce_mean(self.critic.Q1(states, actor_actions))

            # Compute actor gradients and update
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            
            # Update the frozen target models using soft target update
            for src, tgt in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
                tgt.assign(self.tau * src + (1.0 - self.tau) * tgt)

            for src, tgt in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
                tgt.assign(self.tau * src + (1.0 - self.tau) * tgt)

    def save(self, filename):
        self.actor.save_weights(filename + "_actor")
        self.critic.save_weights(filename + "_critic")
        self.actor_target.save_weights(filename + "_actor_target")
        self.critic_target.save_weights(filename + "_critic_target")
    
    def load(self, filename):
        self.actor.load_weights(filename + "_actor")
        self.critic.load_weights(filename + "_critic")
        self.actor_target.load_weights(filename + "_actor_target")
        self.critic_target.load_weights(filename + "_critic_target")