from read import *
from classifiers import *
from plotting import *


def main():
    (df_train_features, df_train_labels, label_encoder) = read_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(df_train_features, df_train_labels)

    (X_validation, y_validation, label_encoder_V) = read_validation_data()

    #CLASSIFIERS
    #decision_tree(X_train, X_test, y_train, y_test)
    #naive_bayes(X_train, X_test, y_train, y_test)
    #support_vector_machine(X_train, X_test, y_train, y_test)
    #rf = random_forest(X_train, X_test, y_train, y_test)
    #voting(X_train, X_test, y_train, y_test)
    xgb = xgboost(X_train, X_test, y_train, y_test)

    #PREDICTION AND SAVING
    X_EVALUATE = read_evaluation_data()
    y_predict = xgb.predict(X_EVALUATE)
    y_predict_string_labels = label_encoder_V.inverse_transform(y_predict)
    df_predicted_labels = pd.DataFrame(y_predict_string_labels)
    save_prediction_to_txt(df_predicted_labels)


if __name__ == '__main__':
    main()