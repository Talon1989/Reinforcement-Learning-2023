import numpy as np
import matplotlib.pyplot as plt


def onehot_transformation(y: np.ndarray):
    """
    :param y: 1d numpy.ndarray numeric array
    :return: onehot transformation of given y array
    """
    Y = np.zeros([len(np.unique(y)), y.shape[0]])
    for idx, val in enumerate(y):
        Y[int(val), idx] = 1
    return Y.T


class ReplayBuffer:

    def __init__(self, max_size=1_000):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards)
        return len(self.actions)

    def remember(self, s, a, r, s_, done):
        if len(self.states) > self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.states_[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer(self, batch_size, randomized=True, cleared=False, return_bracket=False):
        assert batch_size <= self.max_size + 1
        indices = np.arange(self.get_buffer_size())
        if randomized:
            np.random.shuffle(indices)
        buffer_states = np.squeeze([self.states[i] for i in indices][0: batch_size])
        buffer_actions = [self.actions[i] for i in indices][0: batch_size]
        buffer_rewards = [self.rewards[i] for i in indices][0: batch_size]
        buffer_states_ = np.squeeze([self.states_[i] for i in indices][0: batch_size])
        buffer_dones = [self.dones[i] for i in indices][0: batch_size]
        if cleared:
            self.clear()
        if return_bracket:
            for i in range(batch_size):
                buffer_actions[i] = np.array(buffer_actions[i])
                buffer_rewards[i] = np.array([buffer_rewards[i]])
                buffer_dones[i] = np.array([buffer_dones[i]])
            return np.array(buffer_states), np.array(buffer_actions), np.array(buffer_rewards), np.array(buffer_states_), np.array(buffer_dones)
            # return tuple(np.array(buffer_states)), tuple(np.array(buffer_actions)), tuple(np.array(buffer_rewards)), tuple(np.array(buffer_states_)), tuple(np.array(buffer_dones))
        return np.array(buffer_states), np.array(buffer_actions), np.array(buffer_rewards), np.array(buffer_states_), np.array(buffer_dones)


class ReplayBufferZeros:

    def __init__(self, max_size, s_dim, a_dim):
        self.states = np.zeros([max_size, s_dim], dtype=np.float32)
        self.actions = np.zeros([max_size, a_dim], dtype=np.float32)
        self.rewards = np.zeros([max_size], dtype=np.float32)
        self.states_ = np.zeros([max_size, s_dim], dtype=np.float32)
        self.dones = np.zeros([max_size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def push(self, s, a, r, s_, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.states_[self.ptr] = s_
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_buffer_size(self):
        return self.size

    def sample(self, batch_size=32, rewards_reshaped=True):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # temp_dict = dict(s=self.states[idxs],
        #                  s2=self.states_[idxs],
        #                  a=self.actions[idxs],
        #                  r=self.rewards[idxs],
        #                  d=self.dones[idxs])
        # print(temp_dict['r'])
        # print()
        # return temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict?['s2'], temp_dict['d']
        if rewards_reshaped:
            return self.states[idxs], self.actions[idxs], self.rewards[idxs].reshape([-1, 1]), self.states_[idxs], self.dones[idxs]
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.states_[idxs], self.dones[idxs]

    def clear(self):
        self.__init__(self.max_size, self.states.shape[1], self.actions.shape[1])


def print_graph(arr1, arr2, label1, label2, title: str, x_label='epoch', scatter=True):
    """
    :param arr1: np.ndarray to be scattered
    :param arr2: np.ndarray to be plotted
    :param label1: label of arr1
    :param label2: label of arr2
    :param title: title of the graph
    :param x_label: label of x-axis
    :param scatter: scatter of plot for the first array
    :return:
    """
    if scatter:
        plt.scatter(np.arange(len(arr1)), arr1, c='g', s=1, label=label1)
    else:
        plt.plot(arr1, c='g', linewidth=1, label=label1)
    plt.plot(arr2, c='b', linewidth=1, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(label1)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    plt.clf()


def print_simple_graph(arr):
    plt.plot(arr, c='b', linewidth=1)
    plt.show()
    plt.clf()


def create_sequential_dataset(dataset, look_back=1):
    """
    :param dataset:
    :param look_back:
    :return: 2 arrays, first is array of sequences of # look_back, second is array of values right after the end of sequences
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_sequential_dataset_2(dataset, look_back=1):
    """
    :param dataset:
    :param look_back:
    :return: 2 arrays, first is array of sequences of # look_back, second is array of values right after the end of sequences
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i: i+look_back])
        dataY.append(dataset[i+look_back])
    return dataX, dataY