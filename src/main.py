"""Implement RL Algorithm for Delivery Route Optimization"""

import collections

from env import DeliveryRouteEnv

GAMMA = 0.9
TEST_EPISODES = 20


def random_play(environment, episodes=10, render=False):

    for ep in range(episodes):

        total_reward = 0.0
        total_steps = 0
        obs = environment.reset()

        while True:
            action = environment.action_space.sample()
            obs, reward, done, _ = environment.step(action)

            if render:
                env.render()

            total_reward += reward
            total_steps += 1
            if done:
                break

        print("Episode done in {} steps with {:.2f} reward".format(total_steps, total_reward))


class Agent:
    def __init__(self, environment):
        self.env = environment
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * \
                          self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__ == '__main__':
    env = DeliveryRouteEnv()

    random_play(env)

    test_env = env
    agent = Agent(environment=DeliveryRouteEnv())
    # writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = -1000.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        # writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.0:
            print("Solved in %d iterations!" % iter_no)
            break
    # writer.close()




