import  numpy as np
from model.Naive_Bayes_Model import NaiveBayesClassifier
from model.predictor import NaiveBayesPredictor

class Evaluator:
    def __init__(self,model_predictor):
        self.predictor = model_predictor

    def evaluate_accuracy(self,X_text,y_test):
        unique_labels = sorted(y_test.unique())
        if len(unique_labels)!=2:
            raise ValueError( "This evaluation method only supports binary classification")
        positive_label = unique_labels[1]
        negative_label = unique_labels[0]

        TP = TN =FP =FN = 0
        total = len(X_text)
        for i in range(total):
            sample = X_text.iloc[i].to_dict()
            true_label = y_test.iloc[i]
            prob_label = self.predictor.predict(sample)

            if true_label == prob_label and true_label == positive_label:
                TP+=1
            elif true_label == prob_label and true_label == negative_label:
                TN +=1
            elif prob_label == positive_label and true_label == negative_label:
                FP+=1
            elif prob_label == negative_label and true_label == positive_label:
                FN +=1

        return  {
            "positive_label": int(positive_label) if isinstance(positive_label, (np.integer, int)) else str(
                positive_label),
            "negative_label": int(negative_label) if isinstance(negative_label, (np.integer, int)) else str(
                negative_label),
            "TP%": TP/total*100,
            "TN%": TN/total*100,
            "FP%": FP/total*100,
            "FN%": FN/total*100,
            "accuracy":(TP + TN) /total * 100
        }

