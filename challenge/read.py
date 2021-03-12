import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import missingno as msno


def read_file(name='TrainOnMe'):
    df_train = pd.read_csv(f'{name}.csv',index_col=0)
    df_train = df_train.dropna() # Drop rows wit NA values
    df_train_labels = df_train['y'] # Separate df for labels only
    df_train_features = df_train.drop(columns=['y']) # Drop label column
    return (df_train_features, df_train_labels)

def main():
    (df_train_features, df_train_labels) = read_file()


if __name__ == '__main__':
    main()