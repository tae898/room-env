import random
import unittest

import gym

import room_env


class RoomEnv1Test(unittest.TestCase):
    def test_init(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            env = gym.make(
                "RoomEnv-v1", seed=random.randint(0, 10000), des_size=des_size
            )
            (observations, question), info = env.reset()
            del env

    def test_wrong_config0(self) -> None:
        des_size = {
            "foo": {
                "Tae": {
                    "human_location": [["A", 1], ["A", 1]],
                    "object_location": {"laptop": [["lap", 4], ["desk", 1]]},
                },
                "Michael": {
                    "human_location": [["A", 1]],
                    "object_location": {"laptop": [["desk", 4], ["lap", 1]]},
                },
                "Vincent": {
                    "human_location": [["A", 1]],
                    "object_location": {"laptop": [["lap", 1], ["desk", 4]]},
                },
            },
            "bar": {"desk": 1, "A": 9997, "lap": 9998},
            "last_timestep": 1000,
            "semantic_knowledge": {"laptop": "desk"},
        }

        with self.assertRaises(KeyError):
            env = gym.make(
                "RoomEnv-v1", seed=random.randint(0, 10000), des_size=des_size
            )
            (observations, question), info = env.reset()
            del env

    def test_sequence(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            env = gym.make(
                "RoomEnv-v1", seed=random.randint(0, 10000), des_size=des_size
            )
            (observations, question), info = env.reset()

            self.assertEqual(len(env.human_sequence), len(env.question_sequence))

    def test_all(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            env = gym.make("RoomEnv-v1", des_size=des_size)
            (observations, question), info = env.reset()
            while True:
                observations, reward, done, truncated, info = env.step(0)
                if done:
                    break
