import numpy as np

goal_xs = {2: -0.6, 5: 0.0, 8: 0.6}
map_goal_index = {key: i for i, key in enumerate(goal_xs.keys())}
goal_xs = {key: np.array(value, dtype=np.float32) for key, value in goal_xs.items()}
