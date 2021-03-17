import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, randint
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, f1_score, make_scorer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
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
    return best_ccp_alpha

def mlp(X_train, X_test, y_train, y_test):
    print('MLP')
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train, y_train)
    print(cross_val_score(mlp, X_test, y_test, cv=5).mean())

def xgboost(X_train, X_test, y_train, y_test):
    print('XGBOOST')
    xgbc = xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, random_state=42, use_label_encoder=False, verbosity=0)
    xgbc = xgbc.fit(X_train, y_train)
    print(cross_val_score(xgbc, X_test, y_test, cv=10).mean())
    param_grid = [
        {
    'colsample_bytree': [0.3, 0.7],
    'gamma': [0.1, 0.3, 0.5],
    'learning_rate': [0.03, 0.1], # default 0.1 
    'max_depth': [2,4,6], # default 3
    'n_estimators': [100], # default 100
    'subsample': [0.3, 0.4, 0.5]
        }
    ]
    # optimal_params = GridSearchCV(xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, verbosity=0), param_grid, cv=5, scoring=make_scorer(f1_score , average='macro'))
    # optimal_params = optimal_params.fit(X_train, y_train)
    # print(cross_val_score(optimal_params, X_test, y_test, cv=5).mean())
    # print(optimal_params.best_params_)
    return xgbc

def random_forest(X_train, X_test, y_train, y_test):
    print('RANDOM FOREST')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy', max_features='auto', max_depth=8)
    rf.fit(X_train, y_train)
    print(classification_report(y_test, rf.predict(X_test)))
    print(cross_val_score(rf, X_test, y_test, cv=5).mean())
    #model = BaggingClassifier(base_estimator=rf, n_estimators=10)
    #model.fit(X_train, y_train)
    #print(cross_val_score(model, X_test, y_test, cv=5).mean())
    #print(classification_report(y_test, model.predict(X_test)))
    param_grid = [
        {'criterion': ['entropy'],
        'max_features': ['auto', 'log2'],
        'max_depth': [7,8,9]
        }
    ]
    # optimal_params = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    # optimal_params = optimal_params.fit(X_train, y_train)
    # print(cross_val_score(optimal_params, X_test, y_test, cv=5).mean())
    # print(optimal_params.best_params_)
    return rf


def decision_tree(X_train, X_test, y_train, y_test):
    print('DECISION TREE')
    best_ccp_alpha = find_best_alpha(X_train, X_test, y_train, y_test)
    #print(best_ccp_alpha)
    dt = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha)
    dt.fit(X_train, y_train)
    print(cross_val_score(dt, X_test, y_test, cv=5).mean())
    # model = BaggingClassifier(base_estimator=dt, n_estimators=100)
    # model = model.fit(X_train, y_train)
    # print(cross_val_score(model, X_test, y_test, cv=5))
    return dt

def naive_bayes(X_train, X_test, y_train, y_test):
    print('NAIVE BAYES')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print(cross_val_score(gnb, X_test, y_test, cv=5).mean())
    #model = AdaBoostClassifier(base_estimator=gnb, n_estimators=100)
    #model = model.fit(X_train, y_train)
    #print(cross_val_score(model, X_test, y_test, cv=5))
    return gnb

def support_vector_machine(X_train, X_test, y_train, y_test):
    print('SVM')
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    svm = SVC(C=1, kernel='rbf', gamma=0.05, degree=3)
    svm.fit(X_train_scaled, y_train)
    print(cross_val_score(svm, X_test_scaled, y_test, cv=5).mean())
    #model = BaggingClassifier(base_estimator=svm, n_estimators=200)
    #model = model.fit(X_train_scaled, y_train)
    #print(cross_val_score(model, X_test_scaled, y_test, cv=5))
    param_grid = [
        {'C': [1,3,5,10,20,40],
        'gamma': ['scale', 1, 0.5, 0.1, 0.05],
        'degree': [1,2,3,4,5],
        'kernel': ['rbf']}
    ]
    # optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    # optimal_params = optimal_params.fit(X_train_scaled, y_train)
    # print(cross_val_score(optimal_params, X_test_scaled, y_test, cv=5))
    # print(optimal_params.best_params_)
    return svm

def voting(X_train, X_test, y_train, y_test):
    dt = decision_tree(X_train, X_test, y_train, y_test)
    #nb = naive_bayes(X_train, X_test, y_train, y_test)
    rf = random_forest(X_train, X_test, y_train, y_test)
    xgb = xgboost(X_train, X_test, y_train, y_test)
    #svm = support_vector_machine(X_train, X_test, y_train, y_test)
    #ovr = one_vs_rest(X_train, X_test, y_train, y_test)
    voting = VotingClassifier(estimators=[('dt', dt), ('xgb', xgb), ('rf', rf)], voting='hard')
    voting.fit(X_train, y_train)
    print(cross_val_score(voting, X_test, y_test, cv=5).mean())
