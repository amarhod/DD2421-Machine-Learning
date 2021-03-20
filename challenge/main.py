from read import *
from classifiers import *
from plotting import *


def main():
    #Reading, partitioning (into train and test) and downsample data 
    (df, label_encoder) = read_and_clean_data()
    training, validation = partition_training_and_validation(df)
    X, y = split_X_y(training)
    X, y = downsample_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_validation, y_validation = split_X_y(validation)

    xgb = xgboost(X_train, X_test, y_train, y_test)

    #Measuring accuracy of model with testing set
    validation_prediction = xgb.predict(X_validation)
    print(classification_report(validation_prediction, y_validation))
    
    #Predicting and saving the labels for the evaluation set
    X_eval = read_evaluation_data()
    y_predict = xgb.predict(X_eval)
    y_predict = label_encoder.inverse_transform(y_predict) # Reverse transformation frin integer labels back to original string labels 
    save_prediction_to_txt(pd.DataFrame(y_predict))


if __name__ == '__main__':
    main()