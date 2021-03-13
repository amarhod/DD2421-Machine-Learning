import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.feature_selection import VarianceThreshold


#Reads and cleans the data set 
def read_and_clean_data(name='training_partition'):
    df_train = pd.read_csv(f'{name}.csv', index_col=False)
    df_train = df_train.dropna() # Drop rows wit NA values
    df_train = df_train.drop(columns=['Unnamed: 0']) # Drop rows wit NA values
    df_train['x12'] = df_train['x12'].replace(['Flase'], 'False') #Clean up boolean spelling error
    df_train['x6'] = df_train['x6'].replace(['Bayesian Interference'], 'Bayesian Inference') #Clean up Inference spelling error
    labelencoder = LabelEncoder()
    df_train['y'] = labelencoder.fit_transform(df_train['y']) #Encode strings as integers instead
    df_train['x6'] = labelencoder.fit_transform(df_train['x6']) #Encode strings as integers instead
    df_train['x12'] = labelencoder.fit_transform(df_train['x12']) #Encode booleans as integers instead
    cols = [i for i in df_train.columns if i not in ["y"]]
    #Remove outliers
    for col in cols:
        df_train = df_train[((df_train[col] - df_train[col].mean()) / df_train[col].std()).abs() < 3]
    df_train_labels = df_train['y'] # Separate df for labels only
    df_train_features = df_train.drop(columns=['y']) # Drop label column
    return (df_train_features, df_train_labels)

def convert_to_txt(df_train_features, df_train_labels):
    df_train_features = pd.get_dummies(df_train_features, columns=['x6', 'x12'])
    df_train_features.to_csv(r'challenge_X.txt', header=None, index=None, sep=',', mode='a')
    df_train_labels.to_csv(r'challenge_Y.txt', header=None, index=None, sep=',', mode='a')

def new_data(name='TrainOnMe'):
    df_train = pd.read_csv(f'{name}.csv', index_col=False)
    df_train = df_train.dropna() # Drop rows wit NA values
    df_train = df_train.drop(columns=['Unnamed: 0']) # Drop first column with indexing
    df_train['x12'] = df_train['x12'].replace(['Flase'], 'False') #Clean up boolean spelling error
    df_train['x6'] = df_train['x6'].replace(['Bayesian Interference'], 'Bayesian Inference') #Clean up Inference spelling error
    df_train.to_csv("train_cleaned")

def split_data(name='train_cleaned'):
    df_train = pd.read_csv(f'{name}.csv', index_col=False)
    df_train = df_train.drop(columns=['Unnamed: 0'])
    train, validate = \
              np.split(df_train.sample(frac=1, random_state=42), 
                       [int(.8*len(df_train))])
    train.to_csv('training_partition.csv')
    validate.to_csv('validating_partition.csv')
