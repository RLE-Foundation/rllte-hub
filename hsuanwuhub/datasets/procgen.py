from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np


class Procgen(object):
    def __init__(self) -> None:
        file = hf_hub_download(
            repo_id="RLE-Foundation/HsuanwuHub",
            repo_type="dataset",
            filename="procgen_data.json", 
            subfolder="datasets"
        )
        procgen_data = pd.read_json(file)

    def load_scores(self) -> np.ndarray:
        pass

    def load_curves(self) -> np.ndarray:
        pass