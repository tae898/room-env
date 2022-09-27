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
# des-config xxs
# {'mean_rewards_diff': 55.10497010620764,
#  'mean_rewards_episodic': -28.25290500276062,
#  'mean_rewards_semantic': 22.706611596472168,
#  'mean_rewards_random': 10.171308190201021,
#  'mean_rewards_pre_sem': 56.64664170084516,
#  'complexity': 192,
#  'commonsense_prob': 0.5,
#  'maximum_num_locations_per_object': 3,
#  'maximum_num_objects_per_human': 1,
#  'num_humans': 4,
#  'num_total_objects': 1,
#  'maxiumum_days_period': 16,
#  'allow_random_human': True,
#  'allow_random_question': True,
#  'question_prob': 0.2}

# des-config-xs
# {'mean_rewards_diff': 45.648314648314646,
#  'mean_rewards_episodic': -21.052503052503052,
#  'mean_rewards_semantic': 20.963258963258966,
#  'mean_rewards_random': 1.0866910866910857,
#  'mean_rewards_pre_sem': 45.98079698079698,
#  'complexity': 1024,
#  'commonsense_prob': 0.5,
#  'maximum_num_locations_per_object': 4,
#  'maximum_num_objects_per_human': 2,
#  'num_humans': 8,
#  'num_total_objects': 2,
#  'maxiumum_days_period': 8,
#  'allow_random_human': True,
#  'allow_random_question': True,
#  'question_prob': 0.1}

# des-config-s
# {'mean_rewards_diff': 52.660265660265665,
#  'mean_rewards_episodic': -44.40071040071041,
#  'mean_rewards_semantic': 17.911643911643914,
#  'mean_rewards_random': -5.967587967587969,
#  'mean_rewards_pre_sem': 41.84138084138084,
#  'complexity': 2048,
#  'commonsense_prob': 0.3,
#  'maximum_num_locations_per_object': 2,
#  'maximum_num_objects_per_human': 4,
#  'num_humans': 16,
#  'num_total_objects': 4,
#  'maxiumum_days_period': 4,
#  'allow_random_human': True,
#  'allow_random_question': True,
#  'question_prob': 0.1}

# des-config-m
#  {'mean_rewards_diff': 55.688718688718694,
#   'mean_rewards_episodic': -34.995004995005,
#   'mean_rewards_semantic': 23.611943611943612,
#   'mean_rewards_random': 5.556665556665557,
#   'mean_rewards_pre_sem': 53.74658674658675,
#   'complexity': 8192,
#   'commonsense_prob': 0.3,
#   'maximum_num_locations_per_object': 2,
#   'maximum_num_objects_per_human': 4,
#   'num_humans': 32,
#   'num_total_objects': 8,
#   'maxiumum_days_period': 4,
#   'allow_random_human': True,
#   'allow_random_question': True,
#   'question_prob': 0.1},

# des-config-l
# {'mean_rewards_diff': 58.919302919302915,
#  'mean_rewards_episodic': -16.435897435897438,
#  'mean_rewards_semantic': 28.041181041181044,
#  'mean_rewards_random': -16.72882672882673,
#  'mean_rewards_pre_sem': 57.21145521145521,
#  'complexity': 32768,
#  'commonsense_prob': 0.5,
#  'maximum_num_locations_per_object': 2,
#  'maximum_num_objects_per_human': 4,
#  'num_humans': 64,
#  'num_total_objects': 16,
#  'maxiumum_days_period': 4,
#  'allow_random_human': True,
#  'allow_random_question': True,
#  'question_prob': 0.1}
