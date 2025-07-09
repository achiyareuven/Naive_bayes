import pandas as pd
from model.Naive_Bayes_Model import NaiveBayesClassifier
from data_loader.load_data_from_csv import CSVLoader
from model.predictor import NaiveBayesPredictor

from evaluation.evaluator import Evaluator


d = CSVLoader('data/buy_computer_data.csv')
df =d.load_data()
X = df.drop(columns=['buys_computer'])
y = df['buys_computer']

split_index = int(0.7 * len(df))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

model = NaiveBayesClassifier()
model.fit(X_train, y_train)
model.calculate_feature_probs()

predictor = NaiveBayesPredictor(model)

evaluator =Evaluator(predictor)
result = evaluator.evaluate_accuracy(X_test,y_test)

print(result)



#
# df = pd.read_csv('data/buy_computer_data.csv')
#
# X = df.drop(columns=['buys_computer'])
# y = df['buys_computer']
#
# model = NaiveBayesClassifier()
# model.training_model(X, y)
#
# sample = {
#     'age': 'youth',
#     'income': 'medium',
#     'student': 'yes',
#     'credit_rating': 'fair'
# }
#
# print(model.predict(sample))
# print(model.predict_proba(sample))









