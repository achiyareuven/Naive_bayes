import pandas as pd

class DataCleaner:
    def __init__(self,fillna = "unknown"):
        self.fillna_value = fillna

    def clean_data(self,df):
        return df.fillna(self.fillna_value)
