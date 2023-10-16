from typing import Dict
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np


class DM_Control(object):
    def __init__(self) -> None:
        file = hf_hub_download(
            repo_id="RLE-Foundation/HsuanwuHub",
            repo_type="dataset",
            filename="dm_control.json", 
            subfolder="datasets"
        )
        self.dm_control_data = pd.read_json(file)

    def load_scores(self) -> Dict[str, np.ndarray]:
        """Returns final performance"""
        scores_dict = dict()
        for algo in self.dm_control_data.keys():
            scores_dict[algo] = np.array([value for _, value in self.dm_control_data[algo].items()]).T
        
        return scores_dict

    def load_curves(self) -> np.ndarray:
        pass
