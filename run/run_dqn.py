import numpy as np

from env.env import Maze
from agent.DQN_agent import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(100):
        # initial observation
        observation = env.reset()
        rewards = 0
        losses = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            rewards += 1
            transition = np.hstack((observation, action, reward, observation_))
            RL.store_transition(transition)

            if (step > 200) and (step % 5 == 0):
                loss = RL.learn()
                losses += loss

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print(rewards, " ", losses)

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()