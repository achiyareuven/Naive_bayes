#
# from model.Naive_Bayes_Model import NaiveBayesClassifier
# from data_loader.load_data_from_csv import CSVLoader
# from model.predictor import NaiveBayesPredictor
#
# from evaluation.evaluator import Evaluator
#
#
# d = CSVLoader('data/buy_computer_data.csv')
# df =d.load_data()
# X = df.drop(columns=['buys_computer'])
# y = df['buys_computer']
#
# split_index = int(0.7 * len(df))
# X_train = X.iloc[:split_index]
# y_train = y.iloc[:split_index]
#
# X_test = X.iloc[split_index:]
# y_test = y.iloc[split_index:]
#
# model = NaiveBayesClassifier()
# model.fit(X_train, y_train)
#
#
# predictor = NaiveBayesPredictor(model)
#
# evaluator =Evaluator(predictor)
# result = evaluator.evaluate_accuracy(X_test,y_test)
#
# print(result)
# sample = X.iloc[13].to_dict()
# print(df)
#
# print(predictor.predict_proba(sample))



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


from data_loader.load_data_from_csv import CSVLoader
from model.Naive_Bayes_Model import NaiveBayesClassifier
from model.predictor import NaiveBayesPredictor
from evaluation.evaluator import Evaluator
from utils.data_utils import split_train_test,split_feature_target
from data_cleaner.clean_data import DataCleaner
loader = CSVLoader('data/phishing.csv')
df = loader.load_data()

clraner = DataCleaner()
clraner.clean_data(df)
df = df.drop(columns=["Index"])
X,y =  split_feature_target(df)
X_train,X_test,y_train,y_test =split_train_test(X,y)

sample = X.iloc[11049].to_dict()


print(df)
model = NaiveBayesClassifier()
model.fit(X_train, y_train)
predictor = NaiveBayesPredictor(model)
evaluator = Evaluator(predictor)
print(predictor.predict_proba(sample))

accuracy = evaluator.evaluate_accuracy(X_test, y_test)
print("Accuracy:", accuracy)








