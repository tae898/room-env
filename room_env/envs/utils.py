import json
import logging
import os
from typing import Tuple

import yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_json(fname: str) -> dict:
    """Read json"""

    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)

    logging.debug(f"reading json {fullpath} ...")
    with open(fullpath, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    logging.debug(f"writing json {fname} ...")
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)
    logging.debug(f"reading yaml {fullpath} ...")
    with open(fullpath, "r") as stream:
        return yaml.safe_load(stream)


def remove_name(entity: str) -> str:
    """Remove name from the entity.

    Args
    ----
    entity: e.g., Tae's laptop

    Returns
    -------
    e.g., laptop

    """
    return entity.split()[-1]


def split_name_entity(name_entity: str) -> Tuple[str, str]:
    """Separate name and entity from the given string.

    Args
    ----
    name_entity: e.g., "Tae's laptop"

    Returns
    -------
    name: e.g., Tae
    entity: e.g., laptop

    """
    logging.debug(f"spliting name and entity from {name_entity}")
    splitted = name_entity.split()
    assert len(splitted) == 2 and "'" in splitted[0]
    name = splitted[0].split("'")[0]
    entity = splitted[1]

    return name, entity
