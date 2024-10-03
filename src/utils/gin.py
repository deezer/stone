import logging
import os
from typing import Any, Dict, List

import gin  # type: ignore
import numpy as np

logging.basicConfig(level=logging.INFO)


def read_gin_file(gin_file: str) -> List[str]:
    gin_info = []
    if os.path.exists(gin_file):
        with open(gin_file) as f:
            gin_info = f.readlines()
        gin_info = [i.replace("\n", "") for i in gin_info if i != "\n"]
    return gin_info


def parse_gin(gin_file: str) -> List[str]:
    """Parse gin config from --gin_file, --gin_param, and the model directory."""
    print("GIN FILE: {}".format(gin_file))
    gin_info = []
    # Parse gin configs, later calls override earlier ones
    with gin.unlock_config():
        # User gin config and user hyperparameters from flags.
        gin.parse_config_files_and_bindings([gin_file] if gin_file != "" else [], [])
    gin_info = read_gin_file(gin_file)
    for i in gin_info:
        print("\t {}".format(i))
    return gin_info


def get_save_dict(
        ) -> Dict[str, Any]:
    schema = {
        "stone": None,
        "loss_fn": None,
        "optimizer": None,
        "circle_type": None,
        "epoch": 0,
        "gin_info": None,
        "val_loss": np.Inf,
    }
    return schema


def gin_info_list_to_dict(lst: List[str]) -> Dict[str, str]:
    return {i.split(" = ")[0]: i.split(" = ")[1] for i in lst if " = " in i}
