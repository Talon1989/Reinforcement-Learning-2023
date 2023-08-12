import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import utilities
keras = tf.keras
import gym
from utilities import ReplayBuffer
from utilities import ReplayBufferZeros


env_ = gym.make('Pendulum-v1')


class TD3:

    def __init__(self, env, hidden_actor, hidden_critic, alpha=1/1_000, beta=1/500, gamma=99/100,
                 tau=1/100, noise=3, noise_decay=99/100, batch_size=2**6, actor_update_rate=2):
        self.env = env
        self.n_s, self.n_a = env.observation_space.shape[0], env.action_space.shape[0]
        self.low_action, self.high_action = env.action_space.low[0], env.action_space.high[0]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.noise_decay = noise_decay
        self.batch_size = batch_size
        self.actor_update_rate = actor_update_rate
        self.buffer = ReplayBufferZeros(max_size=1_000, s_dim=self.n_s, a_dim=self.n_a)
        self.actor, self.target_actor = self._build_actor_nn(hidden_actor), self._build_actor_nn(hidden_actor)
        self.critic_1, self.critic_2 = self._build_critic_nn(hidden_critic), self._build_critic_nn(hidden_critic)
        self.target_critic_1, self.target_critic_2 = self._build_critic_nn(hidden_critic), self._build_critic_nn(hidden_critic)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha))
        self.critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=self.beta))
        self.critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=self.beta))
        self.critic_loss_function = tf.losses.MeanSquaredError()
        self._soft_update_actor()
        self._soft_update_critic()

    def _build_actor_nn(self, hidden):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='tanh')(layer)
        output_layer = tf.math.multiply(output_layer, self.high_action)
        return keras.models.Model(input_layer, output_layer)

    def _build_critic_nn(self, hidden):
        input_s = keras.layers.Input(shape=self.n_s)
        input_a = keras.layers.Input(shape=self.n_a)
        input_layer = keras.layers.concatenate([input_s, input_a])
        layer = input_layer
        for h in hidden:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        return keras.models.Model([input_s, input_a], output_layer)

    def _soft_update_actor(self, tau=1.):
        self.target_actor.set_weights(
            tau * np.array(self.actor.get_weights(), dtype=object) + (1-tau) * np.array(self.target_actor.get_weights(), dtype=object)
        )

    def _soft_update_critic(self, tau=1.):
        self.target_critic_1.set_weights(
            tau * np.array(self.critic_1.get_weights(), dtype=object) + (1-tau) * np.array(self.target_critic_1.get_weights(), dtype=object)
        )
        self.target_critic_2.set_weights(
            tau * np.array(self.critic_2.get_weights(), dtype=object) + (1-tau) * np.array(self.target_critic_2.get_weights(), dtype=object)
        )

    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        mean = self.actor(s)[0]
        return np.clip(
            np.random.normal(loc=mean, scale=self.noise),
            a_min=self.low_action, a_max=self.high_action
        )

    def _store(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, int(d))

    def _update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            action_values = self.critic_1([states, actions])
            loss = -tf.reduce_mean(action_values)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    def _update_critic(self, states, actions, rewards, states_, dones):
        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
            target_actions = tf.clip_by_value(
                self.target_actor(states_),
                clip_value_min=self.low_action, clip_value_max=self.high_action
            )
            next_state_values = tf.math.minimum(
                self.target_critic_1([states_, target_actions]),
                self.target_critic_2([states_, target_actions])
            )
            y = rewards + self.gamma * next_state_values * (1 - dones)
            critic_1_values = self.critic_1([states, actions])
            critic_2_values = self.critic_2([states, actions])
            loss_1 = self.critic_loss_function(critic_1_values, y)
            loss_2 = self.critic_loss_function(critic_2_values, y)
        grads_1 = tape_1.gradient(loss_1, self.critic_1.trainable_variables)
        grads_2 = tape_2.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))

    def _train(self, update_actor):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        states_ = tf.convert_to_tensor(states_)
        self._update_critic(states, actions, rewards, states_, dones)
        self._soft_update_critic(tau=self.tau)
        if update_actor:
            self._update_actor(states)
            self._soft_update_actor(tau=self.tau)

    def fit(self, n_episodes=2_000, graph=True):
        scores, avg_scores = [], []
        max_steps = self.env._max_episode_steps
        consecutive_solves = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(max_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store(s, a, r, s_, d)
                score += r
                self._train(i % self.actor_update_rate == 0)
                if d or t:
                    if i >= max_steps - 1:
                        consecutive_solves += 1
                    else:
                        consecutive_solves = 0
                    break
                s = s_
            self.noise = self.noise * self.noise_decay
            # if consecutive_solves == 5:
            #     print('Environment solved.')
            #     return
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            print('Episode %d | Noise %.3f | Score %.3f | Avg Score %.3f' % (ep, self.noise, scores[-1], avg_scores[-1]))
            if ep % 20 == 0 and graph:
                utilities.print_graph(scores, avg_scores, 'scores', 'avg scores', 'Ep %d ' % ep)
        return self


class SAC:

    def __init__(
            self, env, hidden_shape_1, hidden_shape_2, hidden_shape_3,
            alpha=1 / 1_000, beta=1 / 1_000, gamma=999 / 1_000, batch_size=2 ** 6
    ):
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.min_action, self.max_action = self.env.action_space.low[0], self.env.action_space.high[0]
        self.gamma = gamma
        self.buffer = ReplayBuffer(max_size=2_000)
        self.batch_size = batch_size
        self.temperature = 1 / 10
        self.ema = tf.train.ExponentialMovingAverage(decay=995 / 1_000)
        self.actor = self._build_actor(hidden_shape_1)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.v, self.target_v = self._build_v(hidden_shape_2), self._build_v(hidden_shape_2)
        self.v.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.q_1 = self._build_q(hidden_shape_3)
        self.q_2 = keras.models.clone_model(self.q_1)
        self.q_1.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.q_2.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.q_and_v_loss = keras.losses.MeanSquaredError()

    def _build_actor(self, hidden):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        mean = keras.layers.Dense(units=self.n_a, activation='linear')(layer)
        log_std = keras.layers.Dense(units=self.n_a, activation='linear')(layer)
        clipped_log_std = tf.clip_by_value(log_std, self.min_action, self.max_action)
        return keras.models.Model(input_layer, [mean, clipped_log_std])

    def _build_v(self, hidden):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        return keras.models.Model(input_layer, output_layer)

    def _build_q(self, hidden):
        state_input = keras.layers.Input(shape=self.n_s)
        action_input = keras.layers.Input(shape=self.n_a)
        layer = keras.layers.concatenate([state_input, action_input])
        for h in hidden:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        return keras.models.Model([state_input, action_input], output_layer)

    def _update_target_v(self):
        self.ema.apply(self.v.trainable_variables)
        for t_v_param, v_param in zip(self.target_v.trainable_variables, self.v.trainable_variables):
            t_v_param.assign(self.ema.average(v_param))

    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        mean, log_std = self.actor(s)
        distribution = tfp.distributions.Normal(loc=mean[0], scale=np.exp(log_std[0]))
        action = np.tanh(distribution.sample())
        return action * self.max_action

    def _store_transition(self, s, a ,r ,s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def _train(self):
        pass
























































































































































































































































































































































































































































































































































































