from collections import defaultdict
import math

from pyexpat import features


class NaiveBayesClassifier:
    def __init__(self):
        self.class_count = defaultdict(int)
        self.features_count = defaultdict(lambda: defaultdict(lambda:defaultdict(int)))
        self.features_values =defaultdict(set)
        self.total_rwos = 0


    def training_model(self,df,target):
        self.total_rwos = len(target)
        for i in range (len(target)):
            label = target.iloc[i]
            self.class_count[label]+=1
            for feature in df.coloms:
                value = df.iloc[i][feature]
                self.features_count[feature][value][label]+=1
                self.features_values[feature].add(value)




