# The Room environment

We have released a challenging [OpenAI Gym](https://gym.openai.com/) compatible environment. The best strategy for this environment is to have both episodic and semantic memory systems. See the [paper](https://arxiv.org/abs/2204.01611) for more information.

This env is added to the PyPI server:

```sh
pip install room-env
```

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

Otherwise, just use what's already there at `room_env/data`

## How does this environment work?

The OpenAI-Gym-compatible Room environment is one big room with
*N*<sub>*people*</sub> number of people who can freely move
around. Each of them selects one object, among
*N*<sub>*objects*</sub>, and places it in one of the
*N*<sub>*locations*</sub> locations.
*N*<sub>*agents*</sub> number of agent(s) are also in this
room. They can only observe one human placing an object, one at a time;
**x**<sup>(*t*)</sup>. At the same time, they are given one question
about the location of an object; **q**<sup>(*t*)</sup>.
**x**<sup>(*t*)</sup> is given as a quadruple,
(**h**<sup>(*t*)</sup>,**r**<sup>(*t*)</sup>,**t**<sup>(*t*)</sup>,*t*),
For example, `<James’s laptop, AtLocation, James’s desk, 42>` accounts
for an observation where an agent sees James placing his laptop on his
desk at *t* = 42. **q**<sup>(*t*)</sup> is given as a double,
(**h**,**r**). For example, `<Karen’s cat, AtLocation>` is asking where
Karen’s cat is located. If the agent answers the question correctly, it
gets a reward of  + 1, and if not, it gets 0.

The reason why the observations and questions are given as
RDF-triple-like format is two folds. One is that this structured format
is easily readable / writable by both humans and machines. Second is
that we can use existing knowledge graphs, such as ConceptNet .

To simplify the environment, the agents themselves are not actually
moving, but the room is continuously changing. There are several random
factors in this environment to be considered:

1. With the chance of *p*<sub>commonsense</sub>,
   a human places an object in a commonsense location (e.g., a laptop
   on a desk). The commonsense knowledge we use is from ConceptNet.
   With the chance of
   1 − *p*<sub>*commonsense*</sub>, an object is
   placed at a non-commonsense random location (e.g., a laptop on the
   tree).

1. With the chance of
   *p*<sub>*new*\_*location*</sub>, a human changes
   object location.

1. With the chance of *p*<sub>*new*\_*object*</sub>, a
   human changes his/her object to another one.

1. With the chance of
   *p*<sub>*switch*\_*person*</sub>, two people
   switch their locations. This is done to mimic an agent moving around
   the room.

All of the four probabilities account for the Bernoulli distributions.

Consider there is only one agent. Then this is a POMDP, where *S*<sub>*t*</sub> =  (**x**<sup>(*t*)</sup>, **q**<sup>(*t*)</sup>), *A*<sub>*t*</sub> = (do something with **x**<sup>(*t*)</sup>, answer **q**<sup>(*t*)</sup>), and *R*<sub>*t*</sub> ∈ *{0, 1}*.

Currently there is no RL trained for this. We only have some heuristics. Take a look at the paper for more details.

## Example

Start with default parameters:

```python
env = gym.make("RoomEnv-v0")
env.reset()
env.step("foo")
```

Start with your own parameters:

```python
env_params = {
    "room_size": "large",
    "weighting_mode": "highest",
    "probs": {
        "commonsense": 0.7,
        "new_location": 0.1,
        "new_object": 0.1,
        "switch_person": 0.5,
    },
    "limits": {
        "heads": None,
        "tails": None,
        "names": None,
        "allow_spaces": False,
    },
    "max_step": 1000,
    "disjoint_entities": True,
    "num_agents": 2,
}
env = gym.make("RoomEnv-v0", **env_params)
env.reset()
env.step("foo")
```

Take a look at [this repo](https://github.com/tae898/explicit-memory) for an actual interaction with this environment.

## TODOs

1. Add unittests.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Cite our work

[![DOI](https://zenodo.org/badge/477781069.svg)](https://zenodo.org/badge/latestdoi/477781069)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
