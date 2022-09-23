from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import pandas as pd
import pyswarms as ps
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time
import datetime

df_liv_1y = pd.read_csv('df_liv_1y_FS.csv',sep=",",error_bad_lines=False, index_col=False, dtype='unicode')
num_var_N = pd.read_csv('num_var_N_FS.csv',sep=",",error_bad_lines=False, index_col=False, dtype='unicode')
cat_var_N = pd.read_csv('cat_var_N_FS.csv',sep=",",error_bad_lines=False, index_col=False, dtype='unicode')



y = df_liv_1y['label_1y']
X = df_liv_1y.drop(['label_1y'],axis=1)

print('Resampled dataset shape %s' % Counter(y))

feature_name = X.columns.tolist()
cat_var_name = cat_var_N.columns.tolist()
num_var_name = num_var_N.columns.tolist()

start = time.time()

# https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/discrete/binary.html
def f_per_particle(m, alpha):
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
       X_subset = X
    else:
       cols = [index for index in range(len(m)) if m[index] == 0]
       X_subset = X.drop(X.columns[cols], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)    
    X_subset_columns = X_subset.columns.tolist()
    X_subset_col_selec =[x for x in X_subset_columns if x in cat_var_name]
    X_train_onehot = pd.get_dummies(X_train, columns = X_subset_col_selec, prefix_sep="_",dummy_na=True)
    X_test_onehot = pd.get_dummies(X_test, columns = X_subset_col_selec, prefix_sep="_",dummy_na=True)
    
    # Remove any columns that aren't in both the training and test sets
    sharedFeatures = set(X_train_onehot.columns) & set(X_test_onehot.columns)
    Remove_train_f = set(X_train_onehot.columns) - sharedFeatures
    Remove_test_f = set(X_test_onehot.columns) - sharedFeatures
    X_train_features = X_train_onehot.drop(list(Remove_train_f), axis=1)
    X_test_features = X_test_onehot.drop(list(Remove_test_f), axis=1)
    
    
    clf = linear_model.LogisticRegression(max_iter=10000)
    clf.fit(X_train_features, y_train)
    predictions = clf.predict(X_test_features)
    P = accuracy_score(y_test, predictions)
#    accuracy = cross_val_score(clf, X_train_features, y_train, cv=10)
    predict_prob_y = clf.predict_proba(X_test_features)
    test_auc = roc_auc_score(y_test,predict_prob_y[:,1])
#    P = (classifier.predict(X_subset) == y).mean()
    j = (alpha * (1.0 - test_auc)
         + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.75):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

pop = 50 # 注意这个100是将n_particles实例化，pop种群数量和n_particles是一样的
iteration = 5 # 注意这个100是循环次数，放这儿是为了显示

dimensions = X.shape[1] # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=50, dimensions=dimensions, options=options)
optimizer.reset()

cost, pos = optimizer.optimize(f, iters=5, verbose=2)

col_f = [index for index in range(len(pos)) if pos[index] == 0]
X_sub = X.drop(X.columns[col_f], axis=1)


df_1 = pd.DataFrame(X_sub)
df_1['label_1y'] = pd.Series(y)


#######################################################################################
# Construct a new model to test it 

y_1y = df_1['label_1y']
X_1y = df_1.drop(['label_1y'],axis=1)


X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_1y, y_1y, test_size=0.3, random_state=42)
X_subset_columns = X_1y.columns.tolist()
X_subset_col_selec =[x for x in X_1y if x in cat_var_name]
X_train_onehot_f = pd.get_dummies(X_train_f, columns = X_subset_col_selec, prefix_sep="_",dummy_na=True)
X_test_onehot_f = pd.get_dummies(X_test_f, columns = X_subset_col_selec, prefix_sep="_",dummy_na=True)

# Remove any columns that aren't in both the training and test sets
sharedFeatures = set(X_train_onehot_f.columns) & set(X_test_onehot_f.columns)
Remove_train_feature = set(X_train_onehot_f.columns) - sharedFeatures
Remove_test_feature = set(X_test_onehot_f.columns) - sharedFeatures
X_train_features_f = X_train_onehot_f.drop(list(Remove_train_feature), axis=1)
X_test_features_f = X_test_onehot_f.drop(list(Remove_test_feature), axis=1)

new_clf = linear_model.LogisticRegression(penalty='l1',solver='liblinear',max_iter=int(1e6))
new_clf.fit(X_train_features_f, y_train_f)
predictions = new_clf.predict(X_test_features_f)
accuracy_f = accuracy_score(y_test_f, predictions)
predict_prob_y = new_clf.predict_proba(X_test_features_f)
auc_f = roc_auc_score(y_test_f,predict_prob_y[:,1])

end = time.time()
run_time = end-start
run_time = str(datetime.timedelta(seconds=run_time))

my_log_lr = open('recode_feature_selection_lr.log', mode = 'a',encoding='utf-8')
print("loss_function：",cost,"选出的特征:",pos,"最终准确率：",accuracy_f,"AUC:",auc_f,"循环次数：",iteration,"种群数量:",pop,"运行时间：",run_time,sep='|',file=my_log_lr)
my_log_lr.close()

df_1.to_csv("df_liv_1y_feature_lr.csv")
