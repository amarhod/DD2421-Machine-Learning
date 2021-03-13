from read import *
from classifiers import *
from plotting import *


def convert_to_txt(df_train_features, df_train_labels):
    df_train_features = pd.get_dummies(df_train_features, columns=['x6', 'x12'])
    df_train_features.to_csv(r'challenge_X.txt', header=None, index=None, sep=',', mode='a')
    df_train_labels.to_csv(r'challenge_Y.txt', header=None, index=None, sep=',', mode='a')

def main():
    (df_train_features, df_train_labels) = read_and_clean_data()
    #decision_tree(df_train_features, df_train_labels)
    #convert_to_txt(df_train_features, df_train_labels)
    naive_bayes(df_train_features, df_train_labels)
    #support_vector_machine(df_train_features, df_train_labels)



if __name__ == '__main__':
    main()