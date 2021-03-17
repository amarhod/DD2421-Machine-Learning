import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, EditedNearestNeighbours, RepeatedEditedNearestNeighbours


#Reads and cleans the data set 
def read_and_clean_data(name='train_cleaned'):
    df_train = pd.read_csv(f'{name}.csv', index_col=False)
    df_train = df_train.dropna() # Drop rows wit NA values
    df_train = df_train.drop(columns=['Unnamed: 0']) # Drop rows wit NA values
    df_train['x12'] = df_train['x12'].replace(['Flase'], 'False') #Clean up boolean spelling error
    df_train['x6'] = df_train['x6'].replace(['Bayesian Interference'], 'Bayesian Inference') #Clean up Inference spelling error
    feature_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    df_train['y'] = label_encoder.fit_transform(df_train['y']) #Encode strings as integers instead
    df_train['x6'] = feature_encoder.fit_transform(df_train['x6']) #Encode strings as integers instead
    df_train['x12'] = feature_encoder.fit_transform(df_train['x12']) #Encode booleans as integers instead
    cols = [i for i in df_train.columns if i not in ["y", 'x6', 'x12']]
    #Remove outliers
    for col in cols:
        df_train = df_train[((df_train[col] - df_train[col].mean()) / df_train[col].std()).abs() < 3]
    y = df_train['y'] # Separate df for labels only
    X = df_train.drop(columns=['y']) # Drop label column
    smote = SMOTE()
    enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority')
    ncr = NeighbourhoodCleaningRule()
    X, y = enn.fit_resample(X,y)
    return (X, y, label_encoder)

def read_validation_data(name='validating_partition'):
    df = pd.read_csv(f'{name}.csv', index_col=False)
    df = df.dropna() # Drop rows wit NA values
    df = df.drop(columns=['Unnamed: 0']) # Drop rows wit NA values
    feature_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    df['y'] = label_encoder.fit_transform(df['y']) #Encode strings as integers instead
    df['x6'] = feature_encoder.fit_transform(df['x6']) #Encode strings as integers instead
    df['x12'] = feature_encoder.fit_transform(df['x12']) #Encode booleans as integers instead
    y = df['y'] # Separate df for labels only
    X = df.drop(columns=['y']) # Drop label column
    return (X, y, label_encoder)

def read_evaluation_data(name='EvaluateOnMe'):
    df = pd.read_csv(f'{name}.csv', index_col=False)
    df = df.dropna() # Drop rows wit NA values
    df = df.drop(columns=['Unnamed: 0']) # Drop rows wit NA values
    feature_encoder = LabelEncoder()
    df['x6'] = feature_encoder.fit_transform(df['x6']) #Encode strings as integers instead
    df['x12'] = feature_encoder.fit_transform(df['x12']) #Encode booleans as integers instead
    return df

def save_prediction_to_txt(df_labels):
    np.savetxt(r'predictionsXGB_FINAL.txt', df_labels.values, fmt='%s')

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
