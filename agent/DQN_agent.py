import numpy as np
import mindspore
from mindspore import context, ops, Tensor, nn
import copy


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class DQN(nn.Cell):
    neuron_nums = 16

    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.net = nn.SequentialCell(
            nn.Dense(n_features, self.neuron_nums),
            nn.Dense(self.neuron_nums, n_actions)
        )

    def construct(self, s):
        print(s)
        print(self.net(s))
        return self.net(s)


class DQNLossCell(nn.Cell):
    def __init__(self, eval_network, target_network, n_features, gamma):
        super(DQNLossCell, self).__init__()
        self.eval_net = eval_network
        self.target_net = target_network
        self.n_features = n_features
        self.gamma = gamma
        self.loss_func = nn.MSELoss()

    def construct(self, batch_memory):
        b_s = batch_memory[:, :self.n_features]
        b_a = ops.ExpandDims()(batch_memory[:, self.n_features], 1).astype(mindspore.int32)
        b_r = ops.ExpandDims()(batch_memory[:, self.n_features + 1], 1)
        b_s_ = batch_memory[:, -self.n_features:]

        q_eval = ops.GatherD()(self.eval_net(b_s), 1, b_a)
        q_next = ops.ReduceMax(keep_dims=True)(self.target_net(b_s_), 1)

        q_target = b_r + self.gamma * q_next
        loss = self.loss_func(q_eval, q_target)

        return loss


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=3,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.eval_net = DQN(self.n_features, self.n_actions)
        self.target_net = copy.deepcopy(self.eval_net)

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        self.opt = nn.Adam(self.eval_net.trainable_params(), learning_rate=self.lr, weight_decay=0.0)

        self.eval_net.set_grad()
        self.dqn_loss_cell = DQNLossCell(self.eval_net, self.target_net, self.n_features, self.gamma)
        self.sens = 1.0
        self.weights = self.opt.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = ops.identity

    def store_transition(self, transition):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def reset_epsilon(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, observation):
        observation = Tensor(observation[np.newaxis, :], mindspore.float32)
        if np.random.uniform() < self.epsilon:
            action_v = self.eval_net(observation)
            action = np.argmax(action_v)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.eval_net.load_parameter_slice(self.target_net.parameters_dict())

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = Tensor(self.memory[sample_index, :], mindspore.float32)
        loss = self.dqn_loss_cell(batch_memory)
        b_s = batch_memory[:, :self.n_features]
        grads = self.grad(self.eval_net, self.weights)(b_s, self.sens)
        grads = self.grad_reducer(grads)
        self.opt(grads)
        print(loss)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1



