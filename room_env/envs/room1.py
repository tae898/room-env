"""Room environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py)
I advise you to check out room2.py, as it's more general.
"""
import logging
import os
import random
from copy import deepcopy
from typing import Tuple

import gym

from ..des import RoomDes

CORRECT = 1
WRONG = 0

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv1(gym.Env):
    """The Room environment version 1"""

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        des_size: str = "l",
        seed: int = 42,
        allow_random_human: bool = False,
        allow_random_question: bool = False,
    ) -> None:
        """Init.

        Args
        ----
        des_size: "xxs", "xs", "s", "m", or "l".
        seed: random seed number
        allow_random_human: whether or not to generate a random human sequence.
        allow_random_question: whether or not to geneate a random question sequence.

        """
        self.seed = seed
        random.seed(self.seed)
        self.allow_random_human = allow_random_human
        self.allow_random_question = allow_random_question

        # Make a dummy observation space.
        self.observation_space = gym.spaces.Discrete(1)
        # 0 for episodic, 1 for semantic, and 2 to forget
        self.action_space = gym.spaces.Discrete(3)

        self.des_size = des_size
        self.des = RoomDes(des_size=self.des_size, check_resources=True, version="v1")

    def generate_sequences(self) -> None:
        """Generate human and question sequences in advance."""
        if self.allow_random_human:
            self.human_sequence = random.choices(
                list(self.des.humans), k=self.des.until + 1
            )
        else:
            self.human_sequence = (
                self.des.humans * (self.des.until // len(self.des.humans) + 1)
            )[: self.des.until + 1]

        if self.allow_random_question:
            self.question_sequence = random.choices(
                list(self.des.humans), k=self.des.until + 1
            )
        else:
            self.question_sequence = [self.human_sequence[0]]
            self.des.run()
            assert (
                len(self.des.states)
                == len(self.des.events) + 1
                == len(self.human_sequence)
            )
            for i in range(len(self.human_sequence) - 1):
                start = max(i + 2 - len(self.des.humans), 0)
                end = i + 2
                humans_observed = self.human_sequence[start:end]

                current_state = self.des.states[end - 1]
                humans_not_changed = []
                for j, human in enumerate(humans_observed):
                    observed_state = self.des.states[start + j]

                    is_changed = False
                    for to_check in ["object", "object_location"]:
                        if (
                            current_state[human][to_check]
                            != observed_state[human][to_check]
                        ):
                            is_changed = True
                    if not is_changed:
                        humans_not_changed.append(human)

                self.question_sequence.append(random.choice(humans_not_changed))

            self.des._initialize()

        assert len(self.human_sequence) == len(self.question_sequence)

    def generate_observation(self) -> dict:
        """Generate an observation.

        Returns
        -------
        observation = {
            "human": <human>,
            "object": <obj>,
            "object_location": <obj_loc>,
        }

        """
        human = self.human_sequence.pop(0)
        obj = self.des.state[human]["object"]
        obj_loc = self.des.state[human]["object_location"]
        observation = deepcopy(
            {
                "human": human,
                "object": obj,
                "object_location": obj_loc,
                "current_time": self.des.current_time,
            }
        )

        return observation

    def generate_qa(self) -> Tuple[dict, dict]:
        """Generate a question and answer.

        Returns
        -------
        question = {"human": <human>, "object": <obj>}
        answer = {"object_location": <obj_loc>}

        """
        human = self.question_sequence.pop(0)
        obj = self.des.state[human]["object"]
        obj_loc = self.des.state[human]["object_location"]

        question = deepcopy({"human": human, "object": obj})
        answer = deepcopy({"object_location": obj_loc})

        return question, answer

    def reset(self) -> Tuple[dict, dict]:
        """Reset the environment.


        Returns
        -------
        observations, question

        """
        self.des._initialize()
        self.generate_sequences()

        observation = self.generate_observation()
        question, self.answer = self.generate_qa()

        assert len(self.human_sequence) == len(self.question_sequence)

        info = {}

        return (observation, question), info

    def step(self, action: str) -> Tuple[Tuple, int, bool, bool, dict]:
        """An agent takes an action.

        Args
        ----
        action: This is the agent's answer to the previous question.

        """
        if str(action).lower() == self.answer["object_location"].lower():
            reward = CORRECT

        else:
            reward = WRONG

        self.des.step()
        observation = self.generate_observation()
        question, self.answer = self.generate_qa()

        info = {}

        if len(self.human_sequence) == 0:
            assert (len(self.question_sequence) == 0) and (
                self.des.current_time == self.des.until
            )
            done = True
        else:
            done = False

        truncated = False

        return (observation, question), reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
