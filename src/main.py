"""Implement RL Algorithm for Delivery Route Optimization"""

import collections

from tensorboardX import SummaryWriter

from env import DeliveryRouteEnv, MAP

GAMMA = 0.9
ALPHA = 0.8


def random_play(environment, episodes=10, render=False, wait=0):

    for ep in range(episodes):

        total_reward = 0.0
        total_steps = 0
        obs = environment.reset()

        while True:
            action = environment.action_space.sample()
            obs, reward, done, _ = environment.step(action)

            if render:
                env.render(wait=wait)

            total_reward += reward
            total_steps += 1
            if done:
                break

        print("Episode done in {} steps with {:.2f} reward".format(total_steps, total_reward))


class Agent:
    def __init__(self, environment):
        self.env = environment
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env, render=False, wait=0):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render(wait=wait)
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':
    env = DeliveryRouteEnv(MAP)
    random_play(env)

    test_env = env
    agent = Agent(environment=DeliveryRouteEnv(MAP, random_start=True))
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = -float('inf')

    while True:
        iter_no += 1

        # Sample simple action and update Q-value
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        # Test current Q-values
        reward = agent.play_episode(test_env)
        writer.add_scalar("reward", reward, iter_no)

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        if reward > 20.0:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()

    agent.play_episode(test_env, render=True, wait=1)
