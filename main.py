# import dill
# from data_loader.load_data_from_csv import CSVLoader
# from model.Naive_Bayes_Model import NaiveBayesClassifier
# from model.predictor import NaiveBayesPredictor
# from evaluation.evaluator import Evaluator
# from utils.data_utils import split_train_test,split_feature_target
# from data_cleaner.clean_data import DataCleaner
# loader = CSVLoader('data/phishing.csv')
# df = loader.load_data()
#
# clraner = DataCleaner()
# clraner.clean_data(df)
# df = df.drop(columns=["Index"])
# X,y =  split_feature_target(df)
# X_train,X_test,y_train,y_test =split_train_test(X,y)
#
# # sample = X.iloc[11049].to_dict()
#
#
# # print(df)
# model = NaiveBayesClassifier()
# model.fit(X_train, y_train)
# predictor = NaiveBayesPredictor(model)
# evaluator = Evaluator(predictor)
# # print(predictor.predict_proba(sample))
# accuracy = evaluator.evaluate_accuracy(X_test, y_test)
#
# with open("trained_model.pkl","wb") as f:
#     dill.dump((model,accuracy),f)
#
# print("✅ Model saved to trained_model.pkl")

from manager import show_menu

if __name__ == "__main__":
    show_menu()








