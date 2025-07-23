# import dill
# import os
# from data_loader.load_data_from_csv import CSVLoader
# from model.Naive_Bayes_Model import NaiveBayesClassifier
# from model.predictor import NaiveBayesPredictor
# from evaluation.evaluator import Evaluator
# from utils.data_utils import split_train_test,split_feature_target
# from data_cleaner.clean_data import DataCleaner
#
# current_accuracy = None
#
# def show_menu():
#     global current_accuracy
#     while True:
#         print("\n========================================================= MODEL MANAGER ==============================================================")
#         print("1. Load data and train model")
#         print("2. show current model accuracy")
#         print("3. exit")
#         choice = input("choos from the list: ").strip()
#
#         match choice:
#             case "1":
#                 file_path = input("enter full file path: ").strip()
#                 try:
#                     df =load_data_by_ending(file_path)
#                     current_accuracy =training_model(df)
#                 except Exception as e:
#                     print(f"failed to load data {e} ")
#             case "2":
#                 if current_accuracy is not None:
#                     print(f"current accuracy is: {current_accuracy}")
#                 elif os.path.exists("trained_model.pkl"):
#                     try:
#                         with open("trained_model.pkl","rb")as f:
#                             model,current_accuracy = dill.load(f)
#                             print(f"current accuracy is: {current_accuracy}")
#                     except Exception as e:
#                         print(f"failed to read file {e}")
#                 else:
#                     print("Model does not exist")
#             case "3":
#                 print("goodby")
#                 break
#             case _:
#                 print("invalid input try again")
#
#
#
#
#
#
# def training_model(df):
#     X, y = split_feature_target(df)
#     X_train, X_test, y_train, y_test = split_train_test(X, y)
#     model = NaiveBayesClassifier()
#     model.fit(X_train, y_train)
#     predictor = NaiveBayesPredictor(model)
#     evaluator = Evaluator(predictor)
#     accuracy = evaluator.evaluate_accuracy(X_test, y_test)
#     try:
#         with open("trained_model.pkl", "wb") as f:
#             dill.dump((model, accuracy), f)
#             print("Model successfully trained and saved to file")
#     except Exception as e:
#         print(f"failed to save model to file {e}")
#     return accuracy
#
#
#
#
#
#
# def load_data_by_ending(file_path):
#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f" File '{file_path}' not found.")
#     ext = os.path.splitext(file_path)[-1].lower()
#     match ext:
#         case ".csv":
#             loader = CSVLoader(file_path)
#             df = loader.load_data()
#
#             cleaner = DataCleaner()
#             df = cleaner.clean_data(df)
#             return df
#         case ".json":
#             raise NotImplementedError("JSON loading not yet implemented.")
#         case _:
#             raise ValueError(f"Unsupported file extension: {ext}")
#
#
#
#
#
#
#
#
#
#
