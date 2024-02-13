import pandas as pd
from datasplit import split_data, split_4_folds

main_df = pd.read_csv("car.data", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
print(main_df.head())

train_data, test_data = split_data(main_df, 0.6)

train_data_1, val_data_1, train_data_2, val_data_2, train_data_3, val_data_3, train_data_4, val_data_4 = split_4_folds(train_data)

