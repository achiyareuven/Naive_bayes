import pandas as pd




def split_feature_target(df:pd.DataFrame,target = None):
    if target == None:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]
    return X,y


def split_train_test(X,y,size = 0.7):
    split_index = int(size * len(X))
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test



