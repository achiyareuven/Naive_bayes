import math

class NaiveBayesPredictor:

    def __init__(self,model):
        self.model= model

    def predict_proba(self, sample):
        probs = {}
        for label in self.model.class_count:
            log_prob = math.log(self.model.class_count[label] / self.model.total_rows)
            for feature in sample:
                value = sample[feature]
                try:
                    prob = self.model.features_probs[feature][value][label]
                except KeyError:
                    prob = 1e-6
                log_prob +=math.log(prob)
            probs[label]=log_prob

        max_log = max(probs.values())
        probs = {label: math.exp(logp - max_log) for label, logp in probs.items()}
        total_prob = sum(probs.values())
        return {label: prob / total_prob for label, prob in probs.items()}

    def predict(self, sample):
        probs = self.predict_proba(sample)
        return max(probs, key=probs.get)
