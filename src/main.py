"""Implement RL Algorithm for Delivery Route Optimization"""

from env import DeliveryRouteEnv

if __name__ == '__main__':
    env = DeliveryRouteEnv()

    for episode in range(20):

        total_reward = 0.0
        total_steps = 0
        obs = env.reset()

        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            env.render()

            total_reward += reward
            total_steps += 1
            if done:
                break

        print("Episode done in {} steps with {:.2f} reward".format(total_steps, total_reward))
