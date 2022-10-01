"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""
import random

from .memory import ShortMemory


def encode_observation(memory_systems: dict, policy: str, obs: dict) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "argmax" or "neural"
    obs: observation = {"human": <human>,
                        "object": <obj>,
                        "object_location": <obj_loc>}

    """
    if policy.lower() == "argmax":
        mem_short = ShortMemory.ob2short(obs)
    else:
        raise NotImplementedError

    memory_systems["short"].add(mem_short)


def manage_memory(memory_systems: dict, policy: str) -> None:
    """Non RL memory management policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic", "semantic", "forget", "random", or "neural"

    """
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "neural",
    ]
    if policy.lower() == "episodic":
        if memory_systems["episodic"].is_full:
            memory_systems["episodic"].forget_oldest()
        mem_short = memory_systems["short"].get_oldest_memory()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems["episodic"].add(mem_epi)

    elif policy.lower() == "semantic":
        if memory_systems["semantic"].is_full:
            memory_systems["semantic"].forget_weakest()
        mem_short = memory_systems["short"].get_oldest_memory()
        mem_sem = ShortMemory.short2sem(mem_short)
        memory_systems["semantic"].add(mem_sem)

    elif policy.lower() == "forget":
        pass

    elif policy.lower() == "random":
        action_number = random.choice([0, 1, 2])
        if action_number == 0:
            if memory_systems["episodic"].is_full:
                memory_systems["episodic"].forget_oldest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems["episodic"].add(mem_epi)

        elif action_number == 1:
            if memory_systems["semantic"].is_full:
                memory_systems["semantic"].forget_weakest()

            mem_short = memory_systems["short"].get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems["semantic"].add(mem_sem)

        else:
            pass

    elif policy.lower() == "neural":
        raise NotImplementedError

    else:
        raise ValueError

    memory_systems["short"].forget_oldest()


def answer_question(memory_systems: dict, policy: str, question: dict) -> str:
    """Non RL question answering policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
            "random", or "neural",
    question: question = {"human": <human>, "object": <obj>}

    Returns
    -------
    pred: prediction

    """
    assert policy.lower() in [
        "episodic_semantic",
        "semantic_episodic",
        "episodic",
        "semantic",
        "random",
        "neural",
    ]
    pred_epi, _ = memory_systems["episodic"].answer_latest(question)
    pred_sem, _ = memory_systems["semantic"].answer_strongest(question)

    if policy.lower() == "episodic_semantic":
        if pred_epi is None:
            pred = pred_sem
        else:
            pred = pred_epi
    elif policy.lower() == "semantic_episodic":
        if pred_sem is None:
            pred = pred_epi
        else:
            pred = pred_sem
    elif policy.lower() == "episodic":
        pred = pred_epi
    elif policy.lower() == "semantic":
        pred = pred_sem
    elif policy.lower() == "random":
        pred = random.choice([pred_epi, pred_sem])
    elif policy.lower() == "neural":
        raise NotImplementedError
    else:
        raise ValueError

    return pred


#### PREDEFINED DES CONFIGS ####
# capacity:
#   episodic: 1
#   semantic: 1
#   short: 1
# des-config xxs
# {
#     "mean_rewards_diff": 8.266666666666666,
#     "mean_rewards_episodic": -1.6,
#     "mean_rewards_semantic": 7.6,
#     "mean_rewards_random": 1.0,
#     "mean_rewards_pre_sem": 10.6,
#     "complexity": 128,
#     "commonsense_prob": 0.5,
#     "maximum_num_locations_per_object": 2,
#     "maximum_num_objects_per_human": 1,
#     "num_humans": 4,
#     "num_total_objects": 1,
#     "maxiumum_days_period": 16,
#     "allow_random_human": False,
#     "allow_random_question": True,
#     "question_prob": 0.1,
# }


# capacity:
#   episodic: 2
#   semantic: 2
#   short: 1
# des-config-xs
# {
#     "mean_rewards_diff": 7.199999999999999,
#     "mean_rewards_episodic": -4.4,
#     "mean_rewards_semantic": 6.0,
#     "mean_rewards_random": 1.4,
#     "mean_rewards_pre_sem": 8.2,
#     "complexity": 512,
#     "commonsense_prob": 0.5,
#     "maximum_num_locations_per_object": 2,
#     "maximum_num_objects_per_human": 2,
#     "num_humans": 8,
#     "num_total_objects": 2,
#     "maxiumum_days_period": 8,
#     "allow_random_human": False,
#     "allow_random_question": True,
#     "question_prob": 0.1,
# }


# capacity:
#   episodic: 4
#   semantic: 4
#   short: 1
# des-config-s
# {
#     "mean_rewards_diff": 5.2,
#     "mean_rewards_episodic": 0.2,
#     "mean_rewards_semantic": 4.2,
#     "mean_rewards_random": 1.0,
#     "mean_rewards_pre_sem": 7.0,
#     "complexity": 1536,
#     "commonsense_prob": 0.5,
#     "maximum_num_locations_per_object": 3,
#     "maximum_num_objects_per_human": 2,
#     "num_humans": 16,
#     "num_total_objects": 4,
#     "maxiumum_days_period": 4,
#     "allow_random_human": False,
#     "allow_random_question": True,
#     "question_prob": 0.1,
# }


# capacity:
#   episodic: 8
#   semantic: 8
#   short: 1
# des-config-m
# {
#     "mean_rewards_diff": 7.2,
#     "mean_rewards_episodic": -5.8,
#     "mean_rewards_semantic": 4.6,
#     "mean_rewards_random": 1.2,
#     "mean_rewards_pre_sem": 7.2,
#     "complexity": 16384,
#     "commonsense_prob": 0.5,
#     "maximum_num_locations_per_object": 2,
#     "maximum_num_objects_per_human": 4,
#     "num_humans": 32,
#     "num_total_objects": 8,
#     "maxiumum_days_period": 8,
#     "allow_random_human": False,
#     "allow_random_question": True,
#     "question_prob": 0.1,
# }

# capacity:
#   episodic: 16
#   semantic: 16
#   short: 1
# des-config-l
# {
#     "mean_rewards_diff": 7.866666666666666,
#     "mean_rewards_episodic": -3.2,
#     "mean_rewards_semantic": 2.8,
#     "mean_rewards_random": -1.6,
#     "mean_rewards_pre_sem": 7.2,
#     "complexity": 98304,
#     "commonsense_prob": 0.5,
#     "maximum_num_locations_per_object": 3,
#     "maximum_num_objects_per_human": 4,
#     "num_humans": 64,
#     "num_total_objects": 16,
#     "maxiumum_days_period": 8,
#     "allow_random_human": False,
#     "allow_random_question": True,
#     "question_prob": 0.1,
# }
