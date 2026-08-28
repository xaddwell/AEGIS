

import os
import json
import numpy as np
import pickle
import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    logger.info(f"Seeding everything with seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def jdump(obj, filename, indent_flag=1):
    directory = os.path.dirname(filename)
    if directory:  # Check if directory is not an empty string
        os.makedirs(directory, exist_ok=True)
    json_dict = json.dumps(obj, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(f"{filename}", 'w') as f:
        if indent_flag:
            json.dump(dict_from_str, f, indent=4)
        else:
            json.dump(dict_from_str, f)

def jload(filename):
    with open(f"{filename}", 'r') as f:
        return json.load(f)
    

def load_pickle(filename):
    with open(f"{filename}.pkl", 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(obj, f)




