# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

import numpy as np

import gym
from gym.spaces import Discrete, Box 


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        np_random = np.random.RandomState()
        random_array = np_random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
        
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)



class MultiSpace(gym.Space):
    """ combining all spaces to a flattened output
        an easier substitute to gym.spaces.Tuple
    """
    def __init__(self, spaces):
        for space in spaces:
            assert isinstance(space, gym.Space), "Elements of the MultiSpace must be instances of gym.Space"
        self.spaces = spaces 
        self.dims = [s.n if isinstance(s, Discrete) else s.shape[0] for s in spaces]
        super(MultiSpace, self).__init__(None, None)

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        """ sample from each space and concatentate: (n,) """
        out = [np.array(s.sample()).reshape(1,-1) for s in self.spaces] # [(1,k)]
        return np.concatenate(out, -1).squeeze()

    def contains(self, x):
        """ x is flattened array or list or tuple """
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        elif isinstance(x, np.ndarray):
            x = self.parse_sample(x)
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space,part) in zip(self.spaces,x))

    def __repr__(self):
        return "MultiSpace(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def parse_sample(self, x):
        """ flatten array x to tuple of samples """
        index, sliced = 0, []
        for dim in self.dims:
            sliced.append(x[index:index+dim])
            index += dim 
        return tuple(sliced)

    def concat_sample(self, x):
        """ combine tuple of samples x to concatenated/flattened array """
        out = [np.array(part).reshape(1,-1) for part in x] # [(1,k)]
        return np.concatenate(out, -1).squeeze()

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        # list of samples, each sample is list of tuples 
        sample_n = [self.parse_sample(s) for s in sample_n]
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        out = [sample for sample in zip(*[space.from_jsonable(sample_n[i]) 
                for i, space in enumerate(self.spaces)])]
        out = [self.concat_sample(s) for s in out]
        return out 

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)
      
    def __eq__(self, other):
        return isinstance(other, MultiSpace) and self.spaces == other.spaces
