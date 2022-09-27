import unittest

import gym

import room_env


class RoomEnv1Test(unittest.TestCase):
    def test_all(self) -> None:
        for room_size in ["small", "big"]:
            env = gym.make("RoomEnv-v0", room_size=room_size)
            observations, info = env.reset()
            while True:
                observations, reward, done, info = env.step("foo")
                if done:
                    break
