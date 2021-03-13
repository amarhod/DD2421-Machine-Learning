import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import plot_confusion_matrix


#Plots a histogram for a given feature
def plot_hist(df, feature='x1'):
    sns.histplot(df[feature])
    plt.show()

#Plots a heatmap for the covariance matrix 
def plot_heat(df, feature='x1'):
    cov_matrix = df.cov()
    plt.rcParams['figure.figsize'] = [len(cov_matrix), len(cov_matrix)]
    sns.heatmap(cov_matrix,
                annot=True,
                cbar = False,
                cmap="YlGnBu",
                xticklabels=range(len(cov_matrix)),
                yticklabels=range(len(cov_matrix)))
    plt.show()

#Plots decision tree
def _plot_tree(dt, df_train_features, df_train_labels):
    plt.figure(figsize=(15,7.5))
    plot_tree(dt,
                filled=True,
                rounded=True,
                class_names=df_train_labels.unique(),
                feature_names=df_train_features.columns)
    plt.show()

#Plots a confusion matrix showing how well a decision tree labeled training set
def plot_tree_confusion(dt, df_train_labels, X_test, y_test):
    plot_confusion_matrix(dt, X_test, y_test, display_labels=df_train_labels.unique())
    plt.show()

#Plots the accuracies for both test and training set as a function of alpha
def plot_tree_alpha_accuracy(ccp_alphas, train_scores, test_scores):
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.plot(ccp_alphas, train_scores, label='training set')
    ax.plot(ccp_alphas, test_scores, label='testing set')
    plt.show()
        

