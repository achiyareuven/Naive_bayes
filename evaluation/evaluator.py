
from model.Naive_Bayes_Model import NaiveBayesClassifier
from model.predictor import NaiveBayesPredictor

class Evaluator:
    def __init__(self,model_predictor):
        self.predictor = model_predictor

    def evaluate_accuracy(self,X_text,y_test):
        correct_count = 0
        total = len(X_text)
        for i in range(total):
            sample = X_text.iloc[i].to_dict()
            true_label = y_test.iloc[i]
            prob_label = self.predictor.predict(sample)

            if true_label == prob_label:
                correct_count +=1
        return  correct_count / total

