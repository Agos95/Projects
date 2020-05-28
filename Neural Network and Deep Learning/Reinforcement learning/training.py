# %% libraries and functions
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt
plt.style.use("mystyle")

def initial_position(x, y, wall):
    pos = [[h,k] for h in range(x) for k in range(y) if [h,k] not in wall]
    choice = np.random.choice(len(pos))
    return pos[choice]

def training(learner, alpha, epsilon, x, y, goal=[0,0], wall=[], sand=[], episodes=2000, episode_length=50, verbose=False):

    # save rewars for each episode
    reward_list = []
    for index in range(0, episodes):
        # start from a random state
        initial = initial_position(x,y,wall)
        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal, wall, sand)
        reward = 0
        # run episode
        for _ in range(0, episode_length):
            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = learner.select_action(state_index, epsilon[index])
            # the agent moves in the environment
            result = env.move(action)
            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
            # update state and reward
            reward += result[1]
            state = result[0]
        reward /= episode_length
        reward_list.append(reward)

        # periodically save the agent
        if verbose and ((index + 1) % 50 == 0):
            """with open('agent.obj', 'wb') as agent_file:
                dill.dump(agent, agent_file)"""
            print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial)
    return learner, reward_list

def test(learner, epsilon, x, y, initial=[0,0], goal=[0,0], wall=[], sand=[], episode_length=50, fname=None):

    # initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal, wall, sand)
    # save positions
    positions = [initial]
    # run episode
    for _ in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon)
        # the agent moves in the environment
        result = env.move(action)
        state = result[0]
        positions.append(list(state))

    env.plot_env(fname, positions)

    return

def plot_reward(reward, fname=None):
    plt.figure()
    plt.plot(reward, ".")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    if fname is not None: plt.savefig(fname)
    plt.close()
    return

# %% main
if __name__ == "__main__":
    episodes = 2000         # number of training episodes
    episode_length = 50     # maximum episode length
    x = 10                  # horizontal size of the box
    y = 10                  # vertical size of the box
    goal = [3, 6]           # objective point
    wall = [
        [0,5], [1,7], [1,8], [1,9], [2,7], [3,1], [3,4], [3,7], [3,8],
        [4,2], [5,7], [6,1], [6,8], [8,2], [8,3], [9,5], [9,8]
    ]
    sand = [
        [0,1], [1,1], [1,2], [1,3], [1,5], [4,8], [5,0], [5,4],
        [5,8], [6,3], [6,4], [7,1], [7,7], [7,8], [8,1], [8,5]
    ]

    # %%

    discount = 0.9          # exponential discount factor
    softmax = True         # set to true to use Softmax policy
    sarsa = True           # set to true to use the Sarsa algorithm

    # TODO alpha and epsilon profile
    alpha = np.ones(episodes) * 0.25
    epsilon = np.linspace(0.8, 0.001, episodes)

    # initialize the agent
    learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

    learner, reward = training(learner, alpha, epsilon, x, y, goal, wall, sand, verbose=True)
    # %%
    plot_reward(reward, fname="reward.pdf")
    # %%
    initial = [6,9]
    test(learner, epsilon[-1], x,y, initial, goal, wall, sand, fname="path.pdf")

