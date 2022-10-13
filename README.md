# The Room environment - v2

For the documentation of [RoomEnv-v0](./documents/README-v0.md) and [RoomEnv-v1](./documents/README-v1.md), click the corresponding buttons.

This document, RoomEnv-v2, is the most up-to-date one.

We have released a challenging [OpenAI Gym](https://www.gymlibrary.dev/) compatible
environment. The best strategy for this environment is to have both episodic and semantic
memory systems. See the [paper](todo/update/the/paper) for more information.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. This env is added to the PyPI server. Just run: `pip install room-env`

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

## [The RoomDes](room_env/des.py)

You can run the RoomDes by

```python
from room_env.des import RoomDes

des = RoomDes()
des.run(debug=True)
```

with `debug=True` it'll print events (i.e., state changes) to the console.

```console
{'resource_changes': {'desk': -1, 'lap': 1},
 'state_changes': {'Vincent': {'current_time': 1,
                               'object_location': {'current': 'desk',
                                                   'previous': 'lap'}}}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {},
 'state_changes': {'Michael': {'current_time': 4,
                               'object_location': {'current': 'lap',
                                                   'previous': 'desk'}},
                   'Tae': {'current_time': 4,
                           'object_location': {'current': 'desk',
                                               'previous': 'lap'}}}}
```

## RoomEnv-v2

```python
import gym
import room_env

env = gym.make("RoomEnv-v2")
observation, info = env.reset()
while True:
    observation, reward, done, truncated, info = env.step(0)
    if done:
        break
```

Every time when an agent takes an action, the environment will give you three memory
systems (i.e., episodic, semantic, and short-term), as an `observation`. The goal of the
agent is to learn a memory management policy. The actions are:

- 0: Put the short-term memory into the epiosdic memory system.
- 1: Put it into the semantic.
- 2: Just forget it.

The memory systems will be managed according to your actions, and they will eventually
used to answer questions. You don't have to worry about the question answering. It's done
by the environment. The better you manage your memory systems, the higher chances that
your agent can answer more questions correctly!

Take a look at [this repo](https://github.com/tae898/explicit-memory) for an actual
interaction with this environment to learn a policy.

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory,
   to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](todo/update/the/paper)

```bibtex
new paper bibtex coming soon
```

## Cite our code

[![DOI](https://zenodo.org/badge/477781069.svg)](https://zenodo.org/badge/latestdoi/477781069)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
