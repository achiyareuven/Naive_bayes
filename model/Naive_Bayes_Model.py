from collections import defaultdict




class NaiveBayesClassifier:
    def __init__(self):
        self.class_count = defaultdict(int)
        self.features_count = defaultdict(lambda: defaultdict(lambda:defaultdict(int)))
        self.features_probs = defaultdict(lambda:defaultdict(lambda :defaultdict(float)))
        self.features_values =defaultdict(set)
        self.total_rows = 0


    def fit(self,X,y):
        self.total_rows = len(y)
        for i in range (len(y)):
            label = y.iloc[i]
            self.class_count[label]+=1
            for feature in X.columns:
                value = X.iloc[i][feature]
                self.features_count[feature][value][label]+=1
                self.features_values[feature].add(value)
        self.calculate_feature_probs()


    def calculate_feature_probs(self):
        for feature in self.features_count:
            for label in self.class_count:
                values = self.features_values[feature]
                total = sum(self.features_count[feature][val][label]+1 for val in values)

                for value in values :
                    count = self.features_count[feature][value][label]+1
                    prob = count/ total
                    self.features_probs[feature][value][label]= prob












