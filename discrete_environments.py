import numpy as np
import tensorflow as tf

import utilities

keras = tf.keras
import gym
from utilities import ReplayBuffer


env_ = gym.make('CartPole-v1')


class DeepQLearning:

    def __init__(self, env, n_hidden, gamma=99/100, alpha=1/1_000, epsilon_decay=9_999/10_000, batch_size=2**6):
        np.random.seed(42)
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.main_nn = self._build_nn(n_hidden)
        self.target_nn = keras.models.clone_model(self.main_nn)
        self.loss_function = keras.losses.MeanSquaredError()
        self.main_nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.alpha)
        )

    def _build_nn(self, n_hidden):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for idx in range(len(n_hidden)):
            layer = keras.layers.Dense(units=n_hidden[idx], activation='relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='linear', dtype=tf.float64)(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def _hard_update(self):
        self.target_nn.set_weights(self.main_nn.get_weights())

    #  hardcoded to work with a single state of gym environments
    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.main_nn(s))

    def _store(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def _train(self):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size, randomized=True, cleared=False
        )
        states = tf.convert_to_tensor(states)
        states_ = tf.convert_to_tensor(states_)
        with tf.GradientTape() as tape:
            y_pred = self.main_nn(states)
            markov_pred = rewards + self.gamma * np.max(self.target_nn(states_), axis=1) * (1 - dones)
            y_hat = np.copy(y_pred)
            y_hat[np.arange(y_pred.shape[0]), actions] = markov_pred
            y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float32)
            loss = self.loss_function(y_pred, y_hat)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.main_nn.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        if self.epsilon > 1/10:
            self.epsilon = self.epsilon * self.epsilon_decay

    def fit(self, n_episodes=5_000, graph=True):
        scores, avg_scores = [], []
        max_steps = self.env._max_episode_steps
        consecutive_solves = 0
        for ep in range(1, n_episodes+1):
            s = self.env.reset()[0]
            score = 0
            for i in range(max_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store(s, a, r, s_, int(d))
                score += r
                self._train()
                if d or t:
                    if i >= max_steps - 1:
                        consecutive_solves += 1
                    else:
                        consecutive_solves = 0
                    break
                s = s_
            if consecutive_solves >= 5:
                print('Environment solved')
                break
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Epsilon %.3f | Avg Score %.3f' % (ep, self.epsilon, avg_scores[-1]))
                self._hard_update()
            if ep % 100 == 0 and graph:
                utilities.print_graph(scores, avg_scores, 'scores', 'avg scores', 'Ep %d ' % ep)
        return self

    def get_some(self):
        for _ in range(5):
            s = self.env.reset()[0]
            a = self.env.action_space.sample()
            s_, r, d, t, _ = self.env.step(a)
            self.buffer.remember(s, a, r, s_, int(d))
        return self.buffer.get_buffer(batch_size=5, randomized=True, cleared=False)


agent = DeepQLearning(env_, [16, 16, 32, 32])
# states, actions, rewards, states_, dones = agent.get_some()
agent.fit()











































































































































































































































































































































































































































































































































