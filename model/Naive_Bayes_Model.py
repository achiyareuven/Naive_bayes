from collections import defaultdict
import math



class NaiveBayesClassifier:
    def __init__(self):
        self.class_count = defaultdict(int)
        self.features_count = defaultdict(lambda: defaultdict(lambda:defaultdict(int)))
        self.features_values =defaultdict(set)
        self.total_rows = 0


    def training_model(self,df,target):
        self.total_rows = len(target)
        for i in range (len(target)):
            label = target.iloc[i]
            self.class_count[label]+=1
            for feature in df.columns:
                value = df.iloc[i][feature]
                self.features_count[feature][value][label]+=1
                self.features_values[feature].add(value)

    def predict_proba(self, sample):
        probs = {}
        for label in self.class_count:
            log_prob = math.log(self.class_count[label] / self.total_rows)
            for feature in sample:
                value = sample[feature]
                value_count = self.features_count[feature][value][label] + 1
                total = sum(self.features_count[feature][v][label] + 1 for v in self.features_values[feature])
                log_prob += math.log(value_count / total)
            probs[label] = log_prob


        max_log = max(probs.values())
        probs = {label: math.exp(logp - max_log) for label, logp in probs.items()}
        total_prob = sum(probs.values())
        return {label: prob / total_prob for label, prob in probs.items()}

    def predict(self, sample):
        probs = self.predict_proba(sample)
        return max(probs, key=probs.get)



