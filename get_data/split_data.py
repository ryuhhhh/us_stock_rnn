"""
csvから以下を取得
 - X_train,y_train
 - X_valid,y_valid
 - X_test,y_test
"""
import numpy as np
import pandas as pd

if __name__ == '__main__':
    csv_name = '1day_data.csv'
    save_name = '1day'
    folder_name = 'not_standard'
    day_1_csv = pd.read_csv(f'./got_data/{csv_name}')

    day_1_csv = day_1_csv.sample(frac=1, random_state=0)
    day_1_csv = day_1_csv.astype({'10per_up':'int8','5per_up':'int8'})

    day_1_csv = round(day_1_csv,3)

    length = len(day_1_csv)

    day_1_test = day_1_csv[:int(length*0.1)]
    day_1_valid = day_1_csv[int(length*0.1):int(length*0.25)]
    day_1_train = day_1_csv[int(length*0.25):]

    day_1_test.to_csv(f'./got_data/data/{folder_name}/{save_name}_test.csv')
    day_1_valid.to_csv(f'./got_data/data/{folder_name}/{save_name}_valid.csv')
    day_1_train.to_csv(f'./got_data/data/{folder_name}/{save_name}_train.csv')
