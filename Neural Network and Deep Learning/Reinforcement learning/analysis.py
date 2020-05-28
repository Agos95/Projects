# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("mystyle")
import seaborn as sns
from training import training, test, plot_reward
import agent
import environment
from pathlib import Path
from tqdm import tqdm
import argparse

# %% functions
def heatmap(df, fname=None):
    df = df.astype("float64")
    plt.figure(figsize=(8,8))
    sns.heatmap(df, vmin=0, vmax=1, linewidths=.05, cbar_kws={"label":"avg reward (last 200 episodes)"})
    plt.ylabel("$\\alpha$")
    plt.xlabel("discount")

    if fname is not None: plt.savefig(fname)
    plt.close()

    return

# %% common parameters

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

epsilon = np.linspace(0.8, 0.001, episodes)

# %% argparser
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="WS", choices=["W", "S", "WS", "No"], help="wether to include Wall and Sand in the environment")
args = parser.parse_args()
if args.env == "W":
    wall += sand
    sand = []
elif args.env == "S":
    sand += wall
    wall = []
elif args.env == "WS":
    pass
else:
    wall = []
    sand = []

env_dir = Path("{}".format(args.env))
env_dir.mkdir(parents=True, exist_ok=True)

# %% plot environment
env = environment.Environment(x, y, [0,0], goal, wall, sand)
env.plot_env(env_dir / "env.pdf")

# %% parameters search
choice        = [True, False]
discount_list = [.1, .3, .6, .9]
a_list        = [.1, .15, .25, .5, .75]
test_initial  = [[6,9], [7,0]]

# counter
cnt = 0
# progress bar
tot = len(choice)*len(choice)*len(discount_list)*len(a_list)
pbar = tqdm(total=tot)
# dataframe to store informations
results = pd.DataFrame(columns=["Softmax", "Sarsa", "Out_Dir", "Discount", "Alpha"])

for softmax in choice:
    for sarsa in choice:
        df = pd.DataFrame(index=a_list, columns=discount_list)
        for discount in discount_list:
            for a in a_list:
                alpha = np.ones(episodes) * a
                # create folder
                cnt += 1
                folder = env_dir / Path("Model{:02d}".format(cnt))
                folder.mkdir(parents=True, exist_ok=True)

                # initialize the agent
                learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
                learner, reward = training(learner, alpha, epsilon, x, y, goal, wall, sand, episodes, episode_length, verbose=False)
                plot_reward(reward, folder / "reward.pdf")
                df.at[a, discount] = np.mean(reward[-200:])
                for i, initial in enumerate(test_initial):
                    test(learner, epsilon[-1], x,y, initial, goal, wall, sand, fname=folder/"path_{}.pdf".format(i))
                # save log in dataframe
                log = pd.DataFrame({
                    "Softmax"  : [softmax],
                    "Sarsa"    : [sarsa],
                    "Out_Dir"  : [folder],
                    "Discount" : [discount],
                    "Alpha"    : [a]
                    })
                results = results.append(log)

                # update progress bar
                pbar.update()

        heatmap(df, fname=env_dir/"softmax_{}-sarsa_{}.pdf".format(softmax, sarsa))

# save results log
results.to_html(env_dir/"summary.html", index=False)
results.to_csv (env_dir/"summary.csv" , index=False)

# close progress bar
pbar.close()
