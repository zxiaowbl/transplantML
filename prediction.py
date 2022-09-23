
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.metrics import roc_curve
from numpy import argmax
import numpy as np
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from MetaCost import MetaCost 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import sklearn
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA


def read_data(file,cat_file,feature_list):
    df_liv_3m = pd.read_csv(file,low_memory=False)
    cat_var = pd.read_csv(cat_file,low_memory=False)
    cat_var_columns = cat_var.columns.tolist()
    with open(feature_list, 'rb') as f:
         feature_list = pickle.load(f)
    y = df_liv_3m['label_1y']
    X = df_liv_3m.drop(['label_1y'],axis=1)

    feature_list= feature_list + ['ALBUMIN_TX','ASCITES_TX','CREAT_TX']
    cat_columns = cat_var_columns + ['ASCITES_TX']
    return X,y,cat_columns, feature_list

# Select the feature by PSO
def PSO_Selection(X_input,f_list):
    X_P = X_input.loc[:,f_list]    
    return X_P


def Split_encoding(X_in,ratio,cat_variables):
    X_train, X_test, y_train, y_test = train_test_split(X_in, y, test_size=ratio, random_state=1)
    X_columns = X_in.columns.tolist()
    X_cat_selec = [x for x in X_columns if x in cat_variables]
    X_train_onehot = pd.get_dummies(X_train, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    X_test_onehot = pd.get_dummies(X_test, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    # Remove any columns that aren't in both the training and test sets
    sharedFeatures = set(X_train_onehot.columns) & set(X_test_onehot.columns)
    Remove_train_f = set(X_train_onehot.columns) - sharedFeatures
    Remove_test_f = set(X_test_onehot.columns) - sharedFeatures
    X_train_features = X_train_onehot.drop(list(Remove_train_f), axis=1)
    X_test_features = X_test_onehot.drop(list(Remove_test_f), axis=1)    
    return X_train_features,X_test_features,y_train,y_test

def encoding(X_in,cat_variables):
    X_columns = X_in.columns.tolist()
    X_cat_selec = [x for x in X_columns if x in cat_variables]
    X_encoding = pd.get_dummies(X_in, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    return X_encoding 

def Smote_process(X,y):
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm,y_sm


def Smote_ENN(X,y):
    sm = SMOTEENN(random_state=42)    
    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm,y_sm

def measure_metrics(model,X_test,y_test):
    y_pred = model.predict(X_test)
    P = accuracy_score(y_test, y_pred)
    predict_prob_y = clf.predict_proba(X_test)
    AUC = roc_auc_score(y_test,predict_prob_y[:,1])
    cnf_matrix = confusion_matrix(y_test, y_pred)
    matrix = classification_report(y_test,y_pred)
    TN = cnf_matrix[0,0]
    FP = cnf_matrix[0,1]
    FN = cnf_matrix[1,0]
    TP = cnf_matrix[1,1]
# https://stats.stackexchange.com/questions/122225/what-is-the-best-way-to-remember-the-difference-between-sensitivity-specificity/212035
# Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
# Specificity or true negative rate
    TNR = TN/(TN+FP) 
# Precision or positive predictive value
    PPV = TP/(TP+FP)
# Negative predictive value
    NPV = TN/(TN+FN)
# Fall out or false positive rate
    FPR = FP/(FP+TN)
# False negative rate
    FNR = FN/(TP+FN)
# False discovery rate
    FDR = FP/(TP+FP)
# Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return TPR,TNR,ACC,AUC,cnf_matrix

def PCA_transform(X_in,n_components):
    pca = PCA(n_components)
    pca.fit(X)
    X_re = pca.transform(X_in)
    return X_re


# Read the dataset
X,y,cat_columns,feature_list = read_data('df_1y_processed.csv','cat_var_New.csv','feature_list.pkl')
# BPSO select the feature
X_PSO = PSO_Selection(X,feature_list)
# train and split the dataset
X_train,X_test,y_train,y_test = Split_encoding(X_PSO,0.3,cat_columns)
# SMOTE for the dataset
#X_sm,y_sm = Smote_process(X_train,y_train)
# Algorithm for the model 

clf = sklearn.tree.DecisionTreeClassifier(max_depth=5,class_weight={0:1,1:1},random_state=0)
#clf = XGBClassifier(silent=1,max_depth=3,learning_rate=0.15,n_estimatores=500,reg_lambda=0.15,use_label_encoder = False,low_memory=False)
#clf.fit(X_sm, y_sm)
clf.fit(X_train, y_train)

#clf.fit(X_train_features, y_train)
TPR,TNR,ACC,AUC,cnf_matrix = measure_metrics(clf,X_test,y_test)

# https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html
def kfold_cv_with_classifier(classifier,
                             X,
                             y,
                             n_splits=10,
                             strategy_name="Basline classifier"):
    
    cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    cv_results_ = sklearn.model_selection.cross_validate(classifier,X,y,cv=cv,
                                                         scoring=['roc_auc','precision','accuracy','f1','recall','neg_brier_score'])                                                                                                                 
    results = round(pd.DataFrame(cv_results_),3)
    results_mean = list(results.mean().values)
    results_std = list(results.std().values)
    results_df = pd.DataFrame([[str(round(results_mean[i],3))+'+/-'+str(round(results_std[i],3)) for i in range(results.shape[1])]],                               
                              columns=['fit_time','score_time','roc_auc','precision','accuracy','f1','recall','brier_score'])                                        
    results_df.rename(index={0:strategy_name}, inplace=True)
    return results_df

def PSO_SMOTEENN(X,y,f_list,cat_variable):
    X_P = PSO_Selection(X,f_list)
    X_s = encoding(X_P,cat_variable)
    X_s, y_s = Smote_ENN(X_s,y)    
    return X_s,y_s    

def PSO_encoding(X,f_list,cat_variable):
    X_P = PSO_Selection(X,f_list)
    X_s = encoding(X_P,cat_variable)   
    return X_s                                                                     


X_t = encoding(X,cat_columns) 
X_s, y_s = Smote_ENN(X_t,y)
results_df = kfold_cv_with_classifier(clf,X_s, y_s, n_splits=10,strategy_name="Decision tree") 

'''
X_s, y_s = Smote_ENN(X_train,y_train)

clf = XGBClassifier(silent=1,max_depth=3,learning_rate=0.15,n_estimatores=500,reg_lambda=0.15,use_label_encoder = False,low_memory=False)

clf.fit(X_train, y_train)

TPR,TNR,ACC,AUC,cnf_matrix = measure_metrics(clf,X_test,y_test)
'''
                                                                                 