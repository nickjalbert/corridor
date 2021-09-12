# Env that models a 1D corridor (you can move left or right)
# Goal is to get to the end (i.e. move right [length] number of times)
import agentos
import numpy as np
from dm_env import specs


# Simulates a 1D corridor
class Corridor(agentos.Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = int(agentos.parameters.corridor_length)
        self.action_space = [0, 1]
        self.observation_space = list(range(self.length + 1))
        self.reset()

    def step(self, action):
        assert action in self.action_space
        if action == 0:
            self.position = np.array(
                [np.float32(max(self.position[0] - 1, 0))]
            )
        else:
            self.position = np.array(
                [np.float32(min(self.position[0] + 1, self.length))]
            )
        return (self.position, np.float32(-1), self.done, dict())

    def reset(self):
        self.position = np.array([np.float32(0)])
        return self.position

    def get_spec(self):
        observations = specs.Array(
            shape=(1,), dtype=np.dtype("float32"), name="observations"
        )
        actions = specs.DiscreteArray(num_values=2, name="actions")
        rewards = specs.Array(
            shape=(), dtype=np.dtype("float32"), name="reward"
        )
        discounts = specs.BoundedArray(
            shape=(),
            dtype=np.dtype("float32"),
            name="discount",
            minimum=0.0,
            maximum=1.0,
        )
        return agentos.EnvironmentSpec(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
        )

    @property
    def valid_actions(self):
        return self.action_space

    @property
    def done(self):
        return self.position[0] >= self.length


# Unit tests for Corridor
def run_tests():
    print("Testing Corridor...")
    agentos.parameters.__dict__["corridor_length"] = 5
    env = Corridor()
    assert env.reset() == 0, "Initial position is 0"
    # Left step in initial position hits a wall and does not change state
    state, reward, done, info = env.step(0)
    assert state == 0
    assert reward == -1
    assert not done
    # Right step should move agent closer to goal
    state, reward, done, info = env.step(1)
    assert state == 1
    assert reward == -1
    assert not done
    # Left step returns agent to initial position
    state, reward, done, info = env.step(0)
    assert state == 0
    assert reward == -1
    assert not done
    # Step to end of corridor
    state, reward, done, info = env.step(1)
    assert state == 1
    assert not done
    state, reward, done, info = env.step(1)
    assert state == 2
    assert not done
    state, reward, done, info = env.step(1)
    assert state == 3
    assert not done
    state, reward, done, info = env.step(1)
    assert state == 4
    assert not done
    state, reward, done, info = env.step(1)
    assert state == 5
    assert done
    print("Tests passed!")


if __name__ == "__main__":
    run_tests()
