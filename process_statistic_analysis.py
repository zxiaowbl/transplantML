# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn import metrics
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
#import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt #MatPlotLib usado para desenhar o gráfico criado com o NetworkX
# Import PySwarms
import pyswarms as ps
from sklearn import linear_model
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy import stats


df_liv_n = pd.read_csv('liver_transplant_data.csv',sep=",",error_bad_lines=False, index_col=False,dtype='unicode')
df_liv_n = df_liv_n.drop(['Unnamed: 0'],axis=1)
# Extract the dataset between 2007/07/01 to 2017/06/30 (transplant date)
df_liv_n['TX_DATE'] = pd.to_datetime(df_liv_n['TX_DATE'])
df_liv=df_liv_n[(df_liv_n['TX_DATE'] >=pd.to_datetime('20030101')) & (df_liv_n['TX_DATE'] <= pd.to_datetime('20121231'))]

# Remove the less than 18
df_liv_1 = df_liv.drop(df_liv[(df_liv['AGE'].astype(float) < 18)].index)
df_liv_t = df_liv[(df_liv['AGE'].astype(float) < 18)]
# Remove the living donor
df_liv_living_donor = df_liv_1[(df_liv_1['DON_TY'] == 'L')]
df_liv_2 = df_liv_1.drop(df_liv_1[(df_liv_1['DON_TY'] == 'L')].index)

# Remove the records that the patient has done re-transplantation after the transplantation
num_liv_PX=df_liv_2['MULTIORG'].value_counts()
df_liv_3=df_liv_2[(df_liv_2['MULTIORG']!='Y')]
# Remove the records that the patient has done re-transplantation
num_liv_PX=df_liv_3['PX_STAT'].value_counts()
df_liv_4=df_liv_3[(df_liv_3['PX_STAT']!='R')]
# Remove the patients lost to follow-up

#%%
#Recipient lost to follow-up within 5-year, n is year
def remove_lost(df,n):
    df.drop(df[(df['PX_STAT']=='L') & (df['GTIME'].astype(float)<90*n)].index,inplace=True)
    df = df.reset_index(drop=True)
    df.drop(df[(df['PSTATUS'].astype(float)==0) & (df['PTIME'].astype(float)< 90*n)].index,inplace=True)
    return df


def calculate_label(df,n):
    df['label_1y'] = 0
    df.label_1y[df['PTIME'].astype(float)<90*n]=1
    return df

df_liv_y1 = remove_lost(df_liv_4,1)
df_liv_y1 = calculate_label(df_liv_y1,1)

#%% Processing the feature
# 5) Remove the feature (time/address)
df_liv_y1 = df_liv_y1.drop(columns=['END_DATE','INIT_DATE','TX_DATE','RECOVERY_DATE_DON','REFERRAL_DATE','ADMISSION_DATE','ADMIT_DATE_DON','PX_STAT_DATE','DISCHARGE_DATE','GRF_FAIL_DATE','PREV_TX_DATE','RETXDATE','COMPOSITE_DEATH_DATE','DEATH_DATE'])
df_liv_y1 = df_liv_y1.drop(columns=['PT_CODE','CTR_CODE','DONOR_ID','END_OPO_CTR_CODE','INIT_OPO_CTR_CODE','LISTING_CTR_CODE','OPO_CTR_CODE','WL_ID_CODE','DISTANCE','TRR_ID_CODE'])
df_liv_y1 = df_liv_y1.drop(columns=['GRF_STAT','GSTATUS','HEP_DENOVO','HEP_RECUR','INFECT','HEPATIC_OUT_OBS','HEPATIC_ART_THROM','OTHER_VASC_THROMB','PORTAL_VEIN_THROM','PRI_GRF_FAIL','PRI_NON_FUNC','PX_NON_COMPL','COD','COD_OSTXT','COD_WL','COD_OSTXT_WL','COD2','COD2_OSTXT','COD3','COD3_OSTXT','BILIARY','FUNC_STAT_TRR'])
df_liv_y1 = df_liv_y1.drop(columns=['ACUTE_REJ_EPI','AGE_GROUP','LOS','LT_ONE_WEEK_DON','DIFFUSE_CHOLANG','PX_STAT','PERM_STATE_TRR','YR_ENTRY_US_TCR','GRF_FAIL_CAUSE_OSTXT','GTIME','RECUR_DISEASE','REJ_ACUTE','REJ_CHRONIC','VASC_THROMB','TRTREJ1Y','TRTREJ6M'])

# Remove the colomn with more than 5% missing values
# Refer to https://xbuba.com/questions/37921703
frac = len(df_liv_y1) * 0.10
df_liv_y1.dropna(thresh=frac, axis=1,inplace=True)

# Distinguish the numerical feature with the categorical feature
df_liv_y1.sort_index(axis=1, inplace=True) # Sort the dataframe according to the dataframe name
df_liv_y1_list = df_liv_y1.columns.tolist()
#variable.sort()
#%%
#df_liv_y1.to_csv(path_or_buf='G:/Submission_code/experiment/1Y_statistic.csv',index=False)

cat_var = pd.read_csv('cat_var_New.csv',low_memory=False)
cat_var_columns = cat_var.columns.tolist()
with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)
feature_list= feature_list + ['ALBUMIN_TX','ASCITES_TX','CREAT_TX']
cat_columns = cat_var_columns + ['ASCITES_TX']


def PSO_Selection(X_input,f_list):
    X_P = X_input.loc[:,f_list]    
    return X_P

def missing (df):
# Calculate the missing value percentage
    missing_number = df.isnull().sum().sort_values(ascending=False) # 每
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Miss number','Miss percentage'])
    return missing_values


df_liv_died = df_liv_y1[(df_liv_y1['label_1y'] == 1)]
df_liv_surv = df_liv_y1[(df_liv_y1['label_1y'] == 0)]

# Statistic analysis for different variable
num_variable = ['AGE','BMI_CALC','ALBUMIN_TX','CREAT_TX','COLD_ISCH','AGE_DON']
Cat_variable = ['GENDER','GENDER_DON','ETHCAT','MED_COND_TRR','HCV_SEROSTATUS','DIAB','MALIG','PORTAL_VEIN_TRR','ETHCAT_DON']

#Black=2 , Hispanic =4

def Calculate_cat_count(df,feature): 
    if feature in Cat_variable:
       Num = df[feature].value_counts() 
       P = Num/df[feature].value_counts().sum()
       print(feature)
       print(Num)
       print(P)
    else:
       print('feature is not in the categorical list')
    return Num,P

def Calculate_num_mean(df,feature): 
    if feature in num_variable:
       mean = df[feature].astype(float).mean()
       std = df[feature].astype(float).std()
       print(feature)
       print(mean)
       print(std)
    else:
        print('feature is not in the numerical list')
    return mean,std

#%% Statistic analysis for categorical variable

Num,P = Calculate_cat_count(df_liv_surv,'GENDER')
Num,P = Calculate_cat_count(df_liv_died,'GENDER')

Num,P = Calculate_cat_count(df_liv_surv,'ETHCAT')
Num,P = Calculate_cat_count(df_liv_died,'ETHCAT')

Num,P = Calculate_cat_count(df_liv_surv,'MED_COND_TRR')
Num,P = Calculate_cat_count(df_liv_died,'MED_COND_TRR')

Num,P = Calculate_cat_count(df_liv_surv,'HCV_SEROSTATUS')
Num,P = Calculate_cat_count(df_liv_died,'HCV_SEROSTATUS')

Num,P = Calculate_cat_count(df_liv_surv,'DIAB')
Num,P = Calculate_cat_count(df_liv_died,'DIAB')

Num,P = Calculate_cat_count(df_liv_surv,'MALIG')
Num,P = Calculate_cat_count(df_liv_died,'MALIG')

Num,P = Calculate_cat_count(df_liv_surv,'PORTAL_VEIN_TRR')
Num,P = Calculate_cat_count(df_liv_died,'PORTAL_VEIN_TRR')

Num,P = Calculate_cat_count(df_liv_surv,'ETHCAT_DON')
Num,P = Calculate_cat_count(df_liv_died,'ETHCAT_DON')

Num,P = Calculate_cat_count(df_liv_surv,'GENDER_DON')
Num,P = Calculate_cat_count(df_liv_died,'GENDER_DON')

# Statistic analysis for the numerical variable

mean,std = Calculate_num_mean(df_liv_surv,'AGE')
mean,std = Calculate_num_mean(df_liv_died,'AGE')

mean,std = Calculate_num_mean(df_liv_surv,'BMI_CALC')
mean,std = Calculate_num_mean(df_liv_died,'BMI_CALC')

mean,std = Calculate_num_mean(df_liv_surv,'ALBUMIN_TX')
mean,std = Calculate_num_mean(df_liv_died,'ALBUMIN_TX')

mean,std = Calculate_num_mean(df_liv_surv,'CREAT_TX')
mean,std = Calculate_num_mean(df_liv_died,'CREAT_TX')

mean,std = Calculate_num_mean(df_liv_surv,'COLD_ISCH')
mean,std = Calculate_num_mean(df_liv_died,'COLD_ISCH')

mean,std = Calculate_num_mean(df_liv_surv,'AGE_DON')
mean,std = Calculate_num_mean(df_liv_died,'AGE_DON')

#%%

# Calculate the chi_square and P value
def chi_square_calculation(df,feature1,feature2):
    c_table = pd.crosstab(df[feature1],df[feature2],margins=True)
    f_obs = np.array([c_table.iloc[0][0:2].values,c_table.iloc[1][0:2].values])
    chi = stats.chi2_contingency(f_obs)[0]
    P = stats.chi2_contingency(f_obs)[1]
    return chi, P
     
chi,P = chi_square_calculation(df_liv_y1,'GENDER','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'ETHCAT','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'MED_COND_TRR','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'HCV_SEROSTATUS','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'DIAB','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'MALIG','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'PORTAL_VEIN_TRR','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'ETHCAT_DON','label_1y')
chi,P = chi_square_calculation(df_liv_y1,'GENDER_DON','label_1y')





