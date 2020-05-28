import numpy as np
import matplotlib.pyplot as plt
plt.style.use("mystyle")
import seaborn as sns

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    wall = []
    sand = []

    def __init__(self, x, y, initial, goal, wall = [], sand = []):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.wall = wall
        self.sand = sand

    # the agent makes an action (0 is stay, 1 is upnum for num in state if num < 0], 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if self.check_boundaries(next_state):
            #print("Boundary:", next_state)
            reward = -1
        elif self.check_wall(next_state):
            #print("Wall:", next_state)
            reward = -1
        elif self.check_sand(next_state):
            #print("Sand:", next_state)
            reward = -.75
            self.state = next_state
        else:
            self.state = next_state
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    def check_wall(self, state):
        state = list(state)
        return True if state in self.wall else False

    def check_sand(self, state):
        state = list(state)
        return True if state in self.sand else False

    def plot_env(self, fname=None, path=None):
        grid = np.zeros(self.boundary)
        for xy in self.wall:
            grid[xy[0], xy[1]] = 2
        for xy in self.sand:
            grid[xy[0], xy[1]] = 3
        grid[self.goal[0], self.goal[1]] = 1

        _, ax = plt.subplots(figsize=(8,8))
        plt.rcParams['figure.autolayout'] = False
        #plt.rcParams['figure.constrained_layout.use'] = True
        #get discrete colormap
        cmap = plt.get_cmap('YlOrRd', 4.)
        # set limits .5 outside true range
        mat = ax.matshow(grid, cmap=cmap, vmin = -.5, vmax = 3.5)
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks(range(10))
        #tell the colorbar to tick at integers
        cax = plt.colorbar(mat, ticks=np.arange(0, 4))
        cax.ax.set_yticklabels(["Path", "Goal", "Wall", "Sand"])
        # plot path if present
        if path is not None:
            path = np.array(path)
            # I need to invert the coordinates, since the coordinates are (x,y),
            # while when referring to an array I have (row, column) -> (y,x)
            plt.plot(path[:,1], path[:,0], "-.", linewidth=2, color="deepskyblue")
            plt.plot(path[0,1], path[0,0], ".", markersize=15, color="navy")
            plt.plot(path[-1,1], path[-1,0], "*", markersize=15, color="navy")
        if fname is not None: plt.savefig(fname)
        plt.close()

        return
