import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from plotting import *

#Finds best alpha for pruning by creating a decision tree for each one and returning the alpha for the one that yielded best accuracy on testing set
def find_best_alpha(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    values = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = values.ccp_alphas
    ccp_alphas = values.ccp_alphas[:-1]
    train_scores = []
    test_scores = []
    for ccp_alpha in ccp_alphas:
        dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        dt = dt.fit(X_train, y_train)
        train_scores.append(np.mean(cross_val_score(dt, X_train, y_train)))
        test_scores.append(np.mean(cross_val_score(dt, X_test, y_test)))

    best_ccp_alpha_index = np.argmax(test_scores)
    best_ccp_alpha = ccp_alphas[best_ccp_alpha_index]
    print(best_ccp_alpha)
    plot_tree_alpha_accuracy(ccp_alphas, train_scores, test_scores)
    return best_ccp_alpha

def random_forest(df_train_features, df_train_labels):
    print("RANDOM FOREST")
    X_train, X_test, y_train, y_test = train_test_split(df_train_features, df_train_labels)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, criterion='entropy', max_features='log2')
    rf = rf.fit(X_train, y_train)
    print(cross_val_score(rf, X_test, y_test, cv=5))

def decision_tree(df_train_features, df_train_labels):
    print("DECISION TREE")
    X_train, X_test, y_train, y_test = train_test_split(df_train_features, df_train_labels)
    best_ccp_alpha = find_best_alpha(X_train, X_test, y_train, y_test)
    dt = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha)
    model = BaggingClassifier(base_estimator=dt, n_estimators=100)
    dt = dt.fit(X_train, y_train)
    print(cross_val_score(dt, X_test, y_test, cv=5))
    model = model.fit(X_train, y_train)
    print(cross_val_score(model, X_test, y_test, cv=5))
    
def naive_bayes(df_train_features, df_train_labels):
    print("NAIVE BAYES")
    X_train, X_test, y_train, y_test = train_test_split(df_train_features, df_train_labels)
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)
    print(cross_val_score(gnb, X_test, y_test, cv=5))
    model = AdaBoostClassifier(base_estimator=gnb, n_estimators=100)
    model = model.fit(X_train, y_train)
    print(cross_val_score(model, X_test, y_test, cv=5))

def support_vector_machine(df_train_features, df_train_labels):
    print("SVM")
    X_train, X_test, y_train, y_test = train_test_split(df_train_features, df_train_labels)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    svm = SVC(C=10, kernel='rbf', gamma='scale')
    svm = svm.fit(X_train_scaled, y_train)
    print(cross_val_score(svm, X_test_scaled, y_test, cv=5))
    model = BaggingClassifier(base_estimator=svm, n_estimators=200)
    model = model.fit(X_train_scaled, y_train)
    print(cross_val_score(model, X_test_scaled, y_test, cv=5))
    param_grid = [
        {'C': [2, 2.2, 2.5, 2.8, 3, 3.2],
        'gamma': ['scale', 1, 0.1, 0.05],
        'kernel': ['rbf']}
    ]
    optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    optimal_params.fit(X_train_scaled, y_train)
    print(cross_val_score(optimal_params, X_test_scaled, y_test, cv=5))
    print(optimal_params.best_params_)
