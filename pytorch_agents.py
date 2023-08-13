import numpy as np
import torch
nn = torch.nn
import utilities
import gym
from utilities import ReplayBuffer
from utilities import ReplayBufferZeros
from utilities import print_graph


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
        self.target_nn = self._build_nn(n_hidden)
        self._hard_update()
        self.loss_function = nn.MSELoss()
        # self.loss_function = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)

    def _build_nn(self, n_hidden):
        model = nn.Sequential()
        for i in range(len(n_hidden)):
            module = nn.Linear(
                in_features=n_hidden[i-1] if i > 0 else self.n_s,
                out_features=n_hidden[i]
            )
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        output_layer = nn.Linear(
            in_features=n_hidden[-1], out_features=self.n_a
        )
        model.add_module(name='l_out', module=output_layer)
        return model

    def _hard_update(self):
        with torch.no_grad():
            self.target_nn.load_state_dict(self.main_nn.state_dict())

    #  hardcoded to work with a single state of gym environments
    def _choose_action(self, s):
        s = torch.from_numpy(s)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return np.argmax(
                    torch.Tensor.numpy(self.main_nn(s).detach())
                )

    def _store(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def _train(self):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size, randomized=True, cleared=False
        )
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions.astype('int64'))
        rewards = torch.from_numpy(rewards)
        states_ = torch.from_numpy(states_)
        dones = torch.from_numpy(dones)
        y_pred = torch.gather(input=self.main_nn(states), dim=1, index=torch.reshape(actions, [-1, 1]))
        y_pred = torch.squeeze(y_pred)
        with torch.no_grad():
            y_pred_next, _ = torch.max(self.target_nn(states_), dim=1)
        y_hat = rewards + self.gamma * y_pred_next * (1 - dones)
        loss = self.loss_function(y_pred, y_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
                score += r
                r = int(r)
                # r = 0 if d == 0 else -100
                self._store(s, a, r, s_, int(d))
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


class PolicyGradient:

    def __init__(self, env, n_hidden, gamma=99/100, alpha=1/1_000):
        np.random.seed(42)
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.main_nn = self._build_nn(n_hidden)
        self.loss_function = self._custom_loss_function
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)

    def _custom_loss_function(self, states, actions, norm_returns):
        distribution = torch.distributions.Categorical(probs=self.main_nn(states))
        log_probability_of_actions = distribution.log_prob(value=actions)
        return - torch.sum(log_probability_of_actions * norm_returns)

    def _build_nn(self, n_hidden):
        model = nn.Sequential()
        for i in range(len(n_hidden)):
            module = nn.Linear(n_hidden[i-1] if i > 0 else self.n_s, n_hidden[i])
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_o', module=nn.Linear(n_hidden[-1], self.n_a))
        model.add_module(name='a_o', module=nn.Softmax(dim=-1))
        return model

    def _choose_action(self, s):
        s = torch.from_numpy(s)
        with torch.no_grad():
            distribution = torch.distributions.Categorical(probs=self.main_nn(s))
        return distribution.sample().item()

    def _store(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _train(self):
        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(), randomized=False, cleared=True
        )
        returns = []
        cumulative = 0
        for r in reversed(range(len(rewards))):
            cumulative = rewards[r] + self.gamma * cumulative
            returns.append(cumulative)
        returns = returns[::-1]
        normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-16)
        losses = self._custom_loss_function(
            torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(normalized_returns)
        )
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def fit(self, n_episodes=5_000, graph=True):
        scores, avg_scores = [], []
        max_steps = self.env._max_episode_steps
        consecutive_solves = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(max_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store(s, a, r)
                score += r
                if d or t:
                    if i >= max_steps - 1:
                        consecutive_solves += 1
                    else:
                        consecutive_solves = 0
                    break
                s = s_
            if consecutive_solves == 5:
                print('Environment solved.')
                return
            self._train()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg Score %.3f' % (ep, avg_scores[-1]))
            if ep % 100 == 0 and graph:
                utilities.print_graph(scores, avg_scores, 'scores', 'avg scores', 'Ep %d ' % ep)
        return self


class ActorCritic:

    def __init__(self, env, hidden_actor, hidden_critic, gamma=99/100, alpha=1/1_000, beta=1/500):
        np.random.seed(42)
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.actor_nn = self._build_actor_nn(hidden_actor)
        self.critic_nn = self._build_critic_nn(hidden_critic)
        self.actor_loss_function = self._custom_actor_loss_function
        self.critic_loss_function = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_nn.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic_nn.parameters(), lr=beta)

    def _custom_actor_loss_function(self, states, actions, returns):
        distribution = torch.distributions.Categorical(probs=self.actor_nn(states))
        log_probability_of_actions = distribution.log_prob(value=actions)
        with torch.no_grad():
            advantages = returns - self.critic_nn(states)
        return - torch.sum(log_probability_of_actions * advantages)

    def _build_actor_nn(self, hidden):
        model = nn.Sequential()
        for i in range(len(hidden)):
            module = nn.Linear(
                in_features=hidden[i-1] if i > 0 else self.n_s,
                out_features=hidden[i]
            )
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(in_features=hidden[-1], out_features=self.n_a))
        model.add_module(name='a_out', module=nn.Softmax(dim=-1))
        return model

    def _build_critic_nn(self, hidden):
        model = nn.Sequential()
        for i in range(len(hidden)):
            module = nn.Linear(
                in_features=hidden[i-1] if i > 0 else self.n_s,
                out_features=hidden[i]
            )
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(in_features=hidden[-1], out_features=1))
        return model

    def _choose_action(self, s):
        s = torch.from_numpy(s)
        with torch.no_grad():
            distribution = torch.distributions.Categorical(probs=self.actor_nn(s))
        return distribution.sample().item()

    def _store(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _train(self):

        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(), randomized=False, cleared=True
        )
        returns = []
        cumulative = 0
        for r in reversed(range(len(rewards))):
            cumulative = rewards[r] + self.gamma * cumulative
            returns.append(cumulative)
        returns = returns[::-1]
        returns = torch.reshape(torch.Tensor(returns), [-1, 1])
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)

        actor_loss = self._custom_actor_loss_function(states, actions, returns)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = self.critic_loss_function(self.critic_nn(states), returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def fit(self, n_episodes=5_000, graph=True):
        scores, avg_scores = [], []
        max_steps = self.env._max_episode_steps
        consecutive_solves = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(max_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store(s, a, r)
                score += r
                if d or t:
                    if i >= max_steps - 1:
                        consecutive_solves += 1
                    else:
                        consecutive_solves = 0
                    break
                s = s_
            if consecutive_solves == 5:
                print('Environment solved.')
                return
            self._train()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg Score %.3f' % (ep, avg_scores[-1]))
            if ep % 100 == 0 and graph:
                utilities.print_graph(scores, avg_scores, 'scores', 'avg scores', 'Ep %d ' % ep)
        return self


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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=beta)
        self.critic_loss_function = nn.MSELoss()
        self._soft_update_actor()
        self._soft_update_critic()

    def _build_actor_nn(self, hidden):
        class Multiply(nn.Module):
            def __init__(self, scalar):
                super().__init__()
                self.scalar = scalar
            def forward(self, x):
                x = torch.mul(x, self.scalar)
                return x
        model = nn.Sequential()
        for i in range(len(hidden)):
            module = nn.Linear(
                in_features=hidden[i-1] if i > 0 else self.n_s,
                out_features=hidden[i]
            )
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(in_features=hidden[-1], out_features=self.n_a))
        model.add_module(name='a_out_1', module=nn.Tanh())
        model.add_module(name='a_out_2', module=Multiply(scalar=self.high_action))
        return model

    def _build_critic_nn(self, hidden):
        model = nn.Sequential()
        for i in range(len(hidden)):
            module = nn.Linear(
                in_features=hidden[i-1] if i > 0 else self.n_s + self.n_a,
                out_features=hidden[i]
            )
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(in_features=hidden[-1], out_features=1))
        return model

    def _soft_update_actor(self, tau=1.):
        for param, t_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

    def _soft_update_critic(self, tau=1.):
        for param, t_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
        for param, t_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

    def _choose_action(self, s):
        s = torch.from_numpy(s)
        with torch.no_grad():
            means = self.actor(s).detach().numpy()
        a = np.random.normal(loc=means, scale=[self.noise for _ in range(self.n_a)])
        return np.clip(a, self.low_action, self.high_action)

    def _store(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, int(d))

    def _update_actor(self, states):
        actions = self.actor(states)
        q_1 = self.critic_1(torch.concat([states, actions], dim=1))
        loss = - torch.mean(q_1)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, states, actions, rewards, states_, dones):

        target_actions = self.target_actor(states_)
        v_ = torch.min(
            self.target_critic_1(torch.concat([states_, target_actions], dim=1)),
            self.target_critic_2(torch.concat([states_, target_actions], dim=1))
        )
        y = rewards + self.gamma * v_ * (1 - dones)

        y_pred_1 = self.critic_1(torch.concat([states, actions], dim=1))
        critic_1_loss = self.critic_loss_function(y_pred_1, y)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()

        y_pred_2 = self.critic_2(torch.concat([states, actions], dim=1))
        critic_2_loss = self.critic_loss_function(y_pred_2, y)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

    def _train(self, update_actor):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.sample(self.batch_size)
        states = torch.Tensor(states)
        if len(actions.shape) == 1:
            actions = torch.Tensor(np.reshape(actions, [-1, 1]))
        else:
            actions = torch.Tensor(actions)
        rewards = torch.Tensor(np.reshape(rewards, [-1, 1]))
        states_ = torch.Tensor(states_)
        dones = torch.Tensor(np.reshape(dones, [-1, 1]))

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


env_ = gym.make('Pendulum-v1')
agent = TD3(env_, [16, 16, 32, 32], [16, 16, 32, 32])
agent.fit()


