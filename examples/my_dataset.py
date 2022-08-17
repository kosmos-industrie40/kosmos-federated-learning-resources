import json
import pandas as pd
from fl_models.abstract.abstract_dataset import AbstractDataset


class MyDataset(AbstractDataset):
    def __init__(self, user_dict=None):
        # Since we can only use keyword arguments, we have to use assertion in order to assure the existence
        assert (
            user_dict is not None
        ), "Please specify a dictionary that should be loaded!"

        self.user_dict = json.loads(user_dict)

    def get_dataframe(self):
        return pd.DataFrame.from_dict(self.user_dict)


def load_class():
    return MyDataset
