# datasplit.py
# contains methods for splitting data into train and test data as well as splitting for k-fold (k=4) cross validation

import pandas as pd

# split_data splits dataframe into 2 where the first has the given percentage of the data and the second has the rest
# data is stratified by class
def split_data(df, percentage):
    result_df_1 = pd.DataFrame(columns=df.columns)
    result_df_2 = pd.DataFrame(columns=df.columns)
    classes = df['class'].drop_duplicates()
    for c in classes:
        c_df = df[df['class'] == c]
        c_df_1 = c_df.sample(frac=percentage, replace=False)
        c_df_2 = c_df[~c_df.index.isin(c_df_1.index)]
        result_df_1 = pd.concat([result_df_1, c_df_1]).reset_index(drop=True)
        result_df_2 = pd.concat([result_df_2, c_df_2]).reset_index(drop=True)

    return result_df_1, result_df_2

# split_4_folds splits dataframe into 4 folds and then compiles 4 training and test sets
# where the test sets are each fold and the training sets are the combined other folds
def split_4_folds(df):
    split_1, split_3 = split_data(df, 0.5)
    split_1, split_2 = split_data(split_1, 0.5)
    split_3, split_4 = split_data(split_3, 0.5)

    train_1 = pd.concat([split_2, split_3, split_4]).reset_index(drop=True)
    train_2 = pd.concat([split_1, split_3, split_4]).reset_index(drop=True)
    train_3 = pd.concat([split_1, split_2, split_4]).reset_index(drop=True)
    train_4 = pd.concat([split_1, split_2, split_3]).reset_index(drop=True)

    return train_1, split_1, train_2, split_2, train_3, split_3, train_4, split_4

# given data that has predicted classes, return the accuracy of the predicted classes to the actual classes
def calculate_accuracy(df):
    total = df.shape[0]
    correct = 0
    for index, row in df.iterrows():
        if row['class'] == row['PredictedClass']:
            correct += 1
    return float(correct) / float(total)
