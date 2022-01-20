import numpy as np
import mindspore
from mindspore import context, ops, Tensor, nn
from mindspore.common.parameter import Parameter, ParameterTuple
import copy


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


_update_op = ops.MultitypeFuncGraph("update_op")


@_update_op.register("Tensor", "Tensor")
def _parameter_update(policy_param, target_param):
    assign = ops.Assign()
    output = assign(target_param, policy_param)
    return output


class DQN(nn.Cell):
    neuron_nums = 16

    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.net = nn.SequentialCell(
            nn.Dense(n_features, self.neuron_nums),
            nn.Dense(self.neuron_nums, n_actions)
        )

    def construct(self, s):
        return self.net(s)


class PolicyNetWithLossCell(nn.Cell):
    """DQN policy network with loss cell"""

    def __init__(self, backbone, loss_fn):
        super(PolicyNetWithLossCell,
              self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.gather = ops.GatherD()

    def construct(self, x, a0, label):
        """constructor for Loss Cell"""
        out = self._backbone(x)
        out = self.gather(out, 1, a0)
        loss = self._loss_fn(out, label)
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
        self.policy_param = ParameterTuple(
            self.eval_net.get_parameters())
        self.target_param = ParameterTuple(
            self.target_net.get_parameters())

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        loss_func = nn.MSELoss()
        opt = nn.Adam(self.eval_net.trainable_params(), learning_rate=self.lr)
        loss_q_net = PolicyNetWithLossCell(self.eval_net, loss_func)
        self.policy_network_train = nn.TrainOneStepCell(loss_q_net, opt)
        self.policy_network_train.set_train(mode=True)

        self.hyper_map = ops.HyperMap()

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

    def update_param(self):
        assign_result = self.hyper_map(
            _update_op,
            self.policy_param,
            self.target_param
        )
        return assign_result

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_param()

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = Tensor(self.memory[sample_index, :], mindspore.float32)
        b_s = batch_memory[:, :self.n_features]
        b_a = ops.ExpandDims()(batch_memory[:, self.n_features], 1).astype(mindspore.int32)
        b_r = ops.ExpandDims()(batch_memory[:, self.n_features + 1], 1)
        b_s_ = batch_memory[:, -self.n_features:]

        q_next = self.target_net(b_s_).max(axis=1)
        q_target = b_r + self.gamma * q_next

        loss = self.policy_network_train(b_s, b_a, q_target)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return loss



