import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb


#Finds best alpha for pruning by creating a decision tree for each one and returning the alpha for the one that yielded best cross val. accuracy on testing set
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

def xgboost(X_train, X_test, y_train, y_test):
    print('XGBOOST')
    xgbc = xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, use_label_encoder=False, verbosity=0)
    xgbc = xgbc.fit(X_train, y_train)
    print('Cross validation:\n', cross_val_score(xgbc, X_test, y_test, cv=5))
    return xgbc

def random_forest(X_train, X_test, y_train, y_test):
    print('RANDOM FOREST')
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', max_depth=8)
    rf.fit(X_train, y_train)
    model = BaggingClassifier(base_estimator=rf, n_estimators=10)
    model.fit(X_train, y_train)
    print('Cross validation:\n', cross_val_score(model, X_test, y_test, cv=5))
    return model

def decision_tree(X_train, X_test, y_train, y_test):
    print('DECISION TREE')
    best_ccp_alpha = find_best_alpha(X_train, X_test, y_train, y_test)
    dt = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha)
    dt.fit(X_train, y_train)
    model = BaggingClassifier(base_estimator=dt, n_estimators=100)
    model = model.fit(X_train, y_train)
    print('Cross validation:\n', cross_val_score(model, X_test, y_test, cv=5))
    return model

def naive_bayes(X_train, X_test, y_train, y_test):
    print('NAIVE BAYES')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    model = AdaBoostClassifier(base_estimator=gnb, n_estimators=100)
    model = model.fit(X_train, y_train)
    print('Cross validation:\n', cross_val_score(model, X_test, y_test, cv=5))
    return model

def support_vector_machine(X_train, X_test, y_train, y_test):
    print('SVM')
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    svm = SVC(C=1, kernel='rbf', gamma=0.05, degree=3)
    svm.fit(X_train_scaled, y_train)
    model = BaggingClassifier(base_estimator=svm, n_estimators=200)
    model = model.fit(X_train_scaled, y_train)
    print('Cross validation:\n', cross_val_score(model, X_test_scaled, y_test, cv=5))
    return model

def voting(X_train, X_test, y_train, y_test):
    print('VOTING')
    dt = decision_tree(X_train, X_test, y_train, y_test)
    rf = random_forest(X_train, X_test, y_train, y_test)
    xgb = xgboost(X_train, X_test, y_train, y_test)
    voting = VotingClassifier(estimators=[('dt', dt), ('xgb', xgb), ('rf', rf)], voting='hard')
    voting.fit(X_train, y_train)
    print('Cross validation:\n', cross_val_score(voting, X_test, y_test, cv=5))
