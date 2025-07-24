import  pandas as pd
from .data_loader_interface import DataLoaderInterFace


class CSVLoader(DataLoaderInterFace):
    def __init__(self,filepath):
        self.filepath = filepath

    def load_data(self):
        return pd.read_csv(self.filepath)
