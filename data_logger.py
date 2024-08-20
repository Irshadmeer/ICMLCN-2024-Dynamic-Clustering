import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class StoreDataCallback(BaseCallback):
    def _on_training_start(self):
        self._log_freq = 10
        self.history_data = []
        
   

    # Define the callback function to append data to the DataFrame
    def _on_step(self) -> bool:
        
        if self.n_calls % self._log_freq == 0:
            _info_dict = self.locals['infos'][0]
            self.history_data.append(_info_dict)   
        return True



