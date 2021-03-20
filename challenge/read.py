import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RepeatedEditedNearestNeighbours


# Reads and cleans the data set 
def read_and_clean_data(name='TrainOnMe'):
    df = pd.read_csv(f'{name}.csv', index_col=False)
    df = df.dropna() # Drop rows wit NA values
    df = df.drop(columns=['Unnamed: 0']) # Drop csv index column
    df['x12'] = df['x12'].replace(['Flase'], 'False') # Clean up boolean spelling error
    df['x6'] = df['x6'].replace(['Bayesian Interference'], 'Bayesian Inference') # Clean up Inference spelling error
    feature_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    df['y'] = label_encoder.fit_transform(df['y']) # Encode labels as integers instead
    df['x6'] = feature_encoder.fit_transform(df['x6']) # Encode strings as integers instead
    df['x12'] = feature_encoder.fit_transform(df['x12']) # Encode booleans as integers instead
    #Remove outliers
    cols = [i for i in df.columns if i not in ['y', 'x6', 'x12']] # Skip
    for col in cols:
        df = df[((df[col] - df[col].mean()) / df[col].std()).abs() < 3]
    return (df, label_encoder)

# Splits the data into label and feature dataframes
def split_X_y(df):
    y = df['y'] # Separate df for labels
    X = df.drop(columns=['y']) # Separate df for features
    return (X, y)

# Partition the training data into a 80/20 split for training and validation 
def partition_training_and_validation(df):
    train, validate = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df))])
    return (train, validate)

# Downsample the data since the class distrobution is skewed
def downsample_data(X,y):
    enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority')
    X, y = enn.fit_resample(X,y)
    return (X, y)

def read_evaluation_data(name='EvaluateOnMe'):
    df = pd.read_csv(f'{name}.csv', index_col=False)
    df = df.dropna() # Drop rows wit NA values
    df = df.drop(columns=['Unnamed: 0']) # Drop rows wit NA values
    feature_encoder = LabelEncoder()
    df['x6'] = feature_encoder.fit_transform(df['x6']) # Encode strings as integers instead
    df['x12'] = feature_encoder.fit_transform(df['x12']) # Encode booleans as integers instead
    return df

# Saves predicted labels for the evaluation set in a text file
def save_prediction_to_txt(df_labels):
    np.savetxt(r'predictions.txt', df_labels.values, fmt='%s')

