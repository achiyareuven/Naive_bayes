import pandas as pd
from sklearn.model_selection import train_test_split




def split_feature_target(df:pd.DataFrame,target = None):
    if target == None:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]
    return X,y


# def split_train_test(X,y,size = 0.7):
#     split_index = int(size * len(X))
#     X_train = X.iloc[:split_index]
#     y_train = y.iloc[:split_index]
#     X_test = X.iloc[split_index:]
#     y_test = y.iloc[split_index:]
#     return X_train, X_test, y_train, y_test



def split_train_test(X, y, size=0.7):
    stratify = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, stratify=stratify, random_state=42
    )
    return X_train, X_test, y_train, y_test




