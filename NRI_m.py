


# Import Modules #
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn import linear_model
import pickle



def bootstrap_results(y_truth, y_pred,num_bootstraps = 1000):
    n_bootstraps = num_bootstraps
    rng_seed = 42  # control reproducibility
    y_pred=y_pred
    y_true=y_truth
    rng = np.random.RandomState(rng_seed)
    tprs=[]
    fprs=[]
    aucs=[]
    threshs=[]
    base_thresh = np.linspace(0, 1, 101)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        fpr, tpr, thresh = metrics.roc_curve(y_true[indices],y_pred[indices])
        thresh=thresh[1:]
        thresh=np.append(thresh,[0.0])
        thresh=thresh[::-1]
        fpr = np.interp(base_thresh, thresh, fpr[::-1])
        tpr = np.interp(base_thresh, thresh, tpr[::-1])
        tprs.append(tpr)
        fprs.append(fpr)
        threshs.append(thresh)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)  
    fprs = np.array(fprs)
    mean_fprs = fprs.mean(axis=0)
    
    return base_thresh, mean_tprs, mean_fprs


def get_auc_ci(y_truth, y_pred,num_bootstraps = 1000):
    n_bootstraps = num_bootstraps
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    y_pred=y_pred
    y_true=y_truth
    rng = np.random.RandomState(rng_seed)
    tprs=[]
    aucs=[]
    base_fpr = np.linspace(0, 1, 101)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        fpr, tpr, _ = metrics.roc_curve(y_true[indices],y_pred[indices])
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    mean_auc = metrics.auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tprs + std*2, 1)
    tprs_lower = mean_tprs - std*2
    return base_fpr, mean_tprs, tprs_lower, tprs_upper, mean_auc, std_auc

def plot_auc(truth, reference_model, new_model,n_bootstraps=1000, save=False):
    y_truth = truth
    ref_model = reference_model
    new_model = new_model
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    ref_auc, new_auc = metrics.auc(ref_fpr, ref_tpr), metrics.auc(new_fpr, new_tpr)
    print('ref auc =',ref_auc, '\nnew auc = ', new_auc)   
    base_fpr_ref, mean_tprs_ref, tprs_lower_ref, tprs_upper_ref, mean_auc_ref, std_auc_ref=get_auc_ci(y_truth, ref_model,n_bootstraps)
    base_fpr_new, mean_tprs_new, tprs_lower_new, tprs_upper_new, mean_auc_new, std_auc_new=get_auc_ci(y_truth, new_model,n_bootstraps)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(ref_fpr, ref_tpr, color='blue',
             lw=lw, label='Reference raw ROC (AUC = %0.2f)' % ref_auc, linestyle='--')
    plt.plot(base_fpr_ref, mean_tprs_ref, 'b', alpha = 0.8, label=r'Reference mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_ref, (mean_auc_ref-2*std_auc_ref),(mean_auc_ref+2*std_auc_ref)),)
    plt.fill_between(base_fpr_ref, tprs_lower_ref, tprs_upper_ref, color = 'b', alpha = 0.2)
    plt.plot(new_fpr, new_tpr, color='darkorange',
             lw=lw, label='New raw ROC (AUC = %0.2f)' % new_auc, linestyle='--')
    plt.plot(base_fpr_new, mean_tprs_new, 'darkorange', alpha = 0.8, label=r'New mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_new,(mean_auc_new-2*std_auc_new),(mean_auc_new+2*std_auc_new)),)
    plt.fill_between(base_fpr_new, tprs_lower_new, tprs_upper_new, color = 'darkorange', alpha = 0.2)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity', fontsize=18)
    plt.ylabel('Sensitivity', fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')


def check_cat(prob,thresholds):
    cat=0
    for i,v in enumerate(thresholds): # 列出四个thresholds, 如果预测的概率大于threshold里面
        if prob>v:                    # 设置了四个thresholds, 找到预测概率大于设定值的索引
            cat=i
    return cat

def make_cat_matrix(ref, new, indices, thresholds):
    num_cats=len(thresholds)
    mat=np.zeros((num_cats,num_cats))   # 建立一个(num_cats,num_cats)的矩阵, (4,4)
    for i in indices: # 每个真实值为0（或1）的索引
        row,col=check_cat(ref[i],thresholds),check_cat(new[i],thresholds)
        mat[row,col]+=1 # 因为初始的threshold建立的是（4,4）矩阵，根据check_cat发现是（3,3），那么（3,3）为1
    return mat
        
def nri(y_truth,y_ref, y_new,risk_thresholds):
    event_index = np.where(y_truth==1)[0]  #y_test里面真实值为1的索引
    nonevent_index = np.where(y_truth==0)[0]  #y_test里面真实值为0的索引
    event_mat=make_cat_matrix(y_ref,y_new,event_index,risk_thresholds) # 对于event_mat， 对于给定的风险值，有多少提升了
    nonevent_mat=make_cat_matrix(y_ref,y_new,nonevent_index,risk_thresholds)
    events_up, events_down = event_mat[0,1:].sum()+event_mat[1,2:].sum()+event_mat[2,3:].sum(),event_mat[1,:1].sum()+event_mat[2,:2].sum()+event_mat[3,:3].sum() #多预测出的正确案例，和没有预测出的正确案例
    nonevents_up, nonevents_down = nonevent_mat[0,1:].sum()+nonevent_mat[1,2:].sum()+nonevent_mat[2,3:].sum(),nonevent_mat[1,:1].sum()+nonevent_mat[2,:2].sum()+nonevent_mat[3,:3].sum() #少预测出的阴性案例，和多预测出的阴性案例
    nri_events = (events_up/len(event_index))-(events_down/len(event_index)) # 算了一个百分比，多预测出的案例占全部案例的百分比减去少预测出的案例占全部案例的百分比
    nri_nonevents = (nonevents_down/len(nonevent_index))-(nonevents_up/len(nonevent_index)) # 多预测出的阴性案例减去少预测出的阴性案例占全部案例的百分比， 最后的NRI相当于将两者加起来
    return nri_events, nri_nonevents, nri_events + nri_nonevents 

def nri_standard_error(num,risk_thresholds,X_ref,y_ref,X_new,y_new,f_list,cat_variables):
    nri_array = []
    nri_eves = []
    nri_none = []
    for i in range(1,num+1):
        X_train, X_test, y_train, y_test = train_test_split(X_new,y_new,test_size=0.3)
        X_ref_train = X_ref.loc[X_train.index.values]
        X_ref_test = X_ref.loc[X_test.index.values]
        y_ref_train = y_ref.loc[y_train.index.values]
        y_ref_test = y_ref.loc[y_test.index.values]     
        X_train_new = PSO_Selection(X_train,f_list)
        X_test_new = PSO_Selection(X_test,f_list)
        X_train_new, X_test_new = dummy_encoding(X_train_new,X_test_new,cat_variables)
        ref_model.fit(X_ref_train.values.reshape(-1, 1), y_ref_train)
        new_model.fit(X_train_new, y_train)
        test_ref_pred=ref_model.predict_proba(X_ref_test.values.reshape(-1, 1))
        test_new_pred=new_model.predict_proba(X_test_new)   
        nri_events,nri_nonevents,nri_overall = nri(y_test.values,test_ref_pred[:,1],test_new_pred[:,1],risk_thresholds)
        nri_array.append(nri_overall)
        nri_eves.append(nri_events)
        nri_none.append(nri_nonevents)        
    return nri_eves,nri_none,nri_array
        
def nri_standard_error_2(num,risk_thresholds,X_ref,y_ref,X_new,y_new):
    nri_array = []
    nri_eves = []
    nri_none = []
    for i in range(1,num+1):
        X_train, X_test, y_train, y_test = train_test_split(X_new,y_new,test_size=0.3)
        X_ref_train = X_ref.loc[X_train.index.values]
        X_ref_test = X_ref.loc[X_test.index.values]
        y_ref_train = y_ref.loc[y_train.index.values]
        y_ref_test = y_ref.loc[y_test.index.values]     
        ref_model.fit(X_ref_train.values.reshape(-1, 1), y_ref_train)
        new_model.fit(X_train.values.reshape(-1, 1), y_train)
        test_ref_pred=ref_model.predict_proba(X_ref_test.values.reshape(-1, 1))
        test_new_pred=new_model.predict_proba(X_test.values.reshape(-1, 1))   
        nri_events,nri_none,nri_overall = nri(y_test.values,test_ref_pred[:,1],test_new_pred[:,1],risk_thresholds)
        nri_array.append(nri_overall)
        nri_eves.append(nri_events)
        nri_none.append(nri_none)                      
    return nri_eves,nri_none,nri_array
        
def track_movement(ref,new, indices):
    up, down = 0,0
    for i in indices:
        ref_val, new_val = ref[i],new[i]
        if ref_val<new_val:
            up+=1
        elif ref_val>new_val:
            down+=1
    return up, down

def category_free_nri(y_truth,y_ref, y_new):
    event_index = np.where(y_truth==1)[0]
    nonevent_index = np.where(y_truth==0)[0]
    events_up, events_down = track_movement(y_ref, y_new,event_index)
    nonevents_up, nonevents_down = track_movement(y_ref, y_new,nonevent_index)
    nri_events = (events_up/len(event_index))-(events_down/len(event_index))
    nri_nonevents = (nonevents_down/len(nonevent_index))-(nonevents_up/len(nonevent_index))
    return nri_events, nri_nonevents, nri_events + nri_nonevents 




def area_between_curves(y1,y2):
    diff = y1 - y2 # calculate difference
    posPart = np.maximum(diff, 0) 
    negPart = -np.minimum(diff, 0) 
    posArea = np.trapz(posPart)
    negArea = np.trapz(negPart)
    return posArea,negArea,posArea-negArea

def plot_idi(y_truth, ref_model, new_model, save=False): 
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    base, mean_tprs, mean_fprs=bootstrap_results( y_truth, new_model,100)
    base2, mean_tprs2, mean_fprs2=bootstrap_results( y_truth, ref_model,100)
    is_pos,is_neg, idi_event=area_between_curves(mean_tprs,mean_tprs2)
    ip_pos,ip_neg, idi_nonevent=area_between_curves(mean_fprs2,mean_fprs)
    print('IS positive', round(is_pos,2),'IS negative',round(is_neg,2),'IDI events',round(idi_event,2))
    print('IP positive', round(ip_pos,2),'IP negative',round(ip_neg,2),'IDI nonevents',round(idi_nonevent,2))
    print('IDI =',round(idi_event+idi_nonevent,2))
    plt.figure(figsize=(10, 10))
    ax=plt.axes()
    lw = 2
    plt.plot(base, mean_tprs, 'black', alpha = 0.5, label='Events New (New)' )
    plt.plot(base, mean_fprs, 'red', alpha = 0.5, label='Nonevents New (New)')
    plt.plot(base2, mean_tprs2, 'black', alpha = 0.7, linestyle='--',label='Events Reference (Ref)' )
    plt.plot(base2, mean_fprs2, 'red', alpha = 0.7,  linestyle='--', label='Nonevents Reference (Ref)')
    plt.fill_between(base, mean_tprs,mean_tprs2, color='black',alpha = 0.1, label='Integrated Sensitivity (area = %0.2f)'%idi_event)
    plt.fill_between(base, mean_fprs,mean_fprs2, color='red', alpha = 0.1, label='Integrated Specificity (area = %0.2f)'%idi_nonevent)
    
    #''' #TODO: comment out if not for breast birads
    ### BIRADS Thresholds ###
    plt.axvline(x=0.02,color='darkorange',linestyle='--',alpha=.5,label='BI-RADS 3/4a Border (2%)')
    plt.axvline(x=0.10,color='green',linestyle='--',alpha=.5,label='BI-RADS 4a/4b Border (10%)')
    plt.axvline(x=0.5,color='blue',linestyle='--',alpha=.5,label='BI-RADS 4b/4c Border (50%)')
    plt.axvline(x=0.95,color='purple',linestyle='--',alpha=.5,label='BI-RADS 4c/5 Border (95%)')
    def nri_annotation(plt, threshold):
        x_pos = base[threshold]
        x_offset=0.02
        x_offset2=x_offset
        text_y_offset=0.01
        text_y_offset2=text_y_offset
        if threshold==2:
            text_y_offset=0.04
            text_y_offset2=0.04
            x_offset2=0.05
            print(x_pos+x_offset, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]])+text_y_offset),
                    x_pos, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]])))
        text_y_events=np.mean([mean_tprs2[threshold], mean_tprs[threshold]])+text_y_offset
        text_y_nonevents=np.mean([mean_fprs[threshold], mean_fprs2[threshold]])+text_y_offset2
        plt.annotate('', xy=(x_pos+0.02, mean_tprs2[threshold+1]), xycoords='data', xytext=(x_pos+0.02, 
                            mean_tprs[threshold]), textcoords='data', arrowprops={'arrowstyle': '|-|'})
        plt.annotate('NRI$_{events}$ = %0.2f'%(mean_tprs[threshold]-mean_tprs2[threshold]), 
                     xy=(x_pos+x_offset, text_y_events), xycoords='data',
                     xytext=(x_pos+x_offset, text_y_events),
                     textcoords='offset points', fontsize=15)
        plt.annotate('', xy=(x_pos+0.02, mean_fprs[threshold]), xycoords='data', xytext=(x_pos+0.02, 
                             mean_fprs2[threshold]), textcoords='data', arrowprops=dict(arrowstyle= '|-|',color='r'))
        plt.annotate('NRI$_{nonevents}$ = %0.2f'%(mean_fprs2[threshold]-mean_fprs[threshold]), 
                     xy=(x_pos+x_offset2, text_y_nonevents), xycoords='data',
                     xytext=(x_pos+x_offset2, text_y_nonevents), 
                     textcoords='offset points', fontsize=15)
        print('Threshold =',round(x_pos,2),'NRI events =',round(mean_tprs[threshold]-mean_tprs2[threshold],4),
              'NRI nonevents =',round(mean_fprs2[threshold]-mean_fprs[threshold],4),'Total =',
              round((mean_tprs[threshold]-mean_tprs2[threshold])+(mean_fprs2[threshold]-mean_fprs[threshold]),4))
    nri_annotation(plt,2)
    nri_annotation(plt,10)
    nri_annotation(plt,50)
    nri_annotation(plt,95)
    #'''
    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Calculated Risk', fontsize=18)
    plt.ylabel('Sensitivity (black), 1 - Specificity (red)', fontsize=18)
    plt.legend(loc="upper right", fontsize=11)
    plt.legend(loc=0, fontsize=11,  bbox_to_anchor=(0,0,1.2,.9))
    plt.gca().set_aspect('equal', adjustable='box')
    #if save:
    #    plt.savefig('idi_curve.png',dpi=300, bbox_inches='tight')
    look=95
    plt.show()

def read_data(file,cat_file,feature_list):
    df_liv_3m = pd.read_csv(file,low_memory=False)
    cat_var = pd.read_csv(cat_file,low_memory=False)
    cat_var_columns = cat_var.columns.tolist()
    with open(feature_list, 'rb') as f:
         feature_list = pickle.load(f)
    y = df_liv_3m['label_1y']
    X = df_liv_3m.drop(['label_1y'],axis=1)
    return X,y,cat_var_columns, feature_list

def Split_encoding(X_in,y_in,ratio,cat_variables):
    X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=ratio, random_state=1)
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

def dummy_encoding(X_train,X_test,cat_variables):
    X_columns = X_train.columns.tolist()
    X_cat_selec = [x for x in X_columns if x in cat_variables]
    X_train_onehot = pd.get_dummies(X_train, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    X_test_onehot = pd.get_dummies(X_test, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    # Remove any columns that aren't in both the training and test sets
    sharedFeatures = set(X_train_onehot.columns) & set(X_test_onehot.columns)
    Remove_train_f = set(X_train_onehot.columns) - sharedFeatures
    Remove_test_f = set(X_test_onehot.columns) - sharedFeatures
    X_train_f = X_train_onehot.drop(list(Remove_train_f), axis=1)
    X_test_f = X_test_onehot.drop(list(Remove_test_f), axis=1)    
    return X_train_f,X_test_f


def encoding(X_in,cat_variables):
    X_columns = X_in.columns.tolist()
    X_cat_selec = [x for x in X_columns if x in cat_variables]
    X_encoding = pd.get_dummies(X_in, columns = X_cat_selec, prefix_sep="|",dummy_na=True)
    return X_encoding     

def PSO_Selection(X_input,f_list):
    X_P = X_input.loc[:,f_list]    
    return X_P

def Softscore(X_input):
    cols = ['AGE','BMI_CALC','NUM_PREV_TX','ALBUMIN_TX','MELD_PELD_LAB_SCORE','ASCITES_TX','MED_COND_TRR','ENCEPH_TX','AGE_DON','SHARE_TY','COLD_ISCH','CREAT_DON','COD_CAD_DON']
    X_P = X_input.loc[:,cols] 
    return X_P

def Barscore(X_input):
    cols = ['AGE','PREV_TX','MELD_PELD_LAB_SCORE','AGE_DON','COLD_ISCH','LIFE_SUP_TRR']
    X_P = X_input.loc[:,cols] 
    return X_P

def Softscore(X_input):
    cols = ['AGE','BMI_CALC','NUM_PREV_TX','ALBUMIN_TX','MELD_PELD_LAB_SCORE','ASCITES_TX','MED_COND_TRR','ENCEPH_TX','AGE_DON','SHARE_TY','COLD_ISCH','CREAT_DON','COD_CAD_DON']
    agepts = np.where(X_input['AGE'] > 60, 4, 0)
    bmipts = np.where(X_input['BMI_CALC'] > 35, 2, 0)
    tranpts = np.where(X_input['NUM_PREV_TX'] == 2.0, 14, (np.where(X_input['NUM_PREV_TX'] == 1.0, 9, 0)))
    #onetranpts = np.where(df_liv_soft['NUM_PREV_TX'] == 1.0, 9, 0)
    #twotranpts = np.where(df_liv_soft['NUM_PREV_TX'] == 2.0, 14, 0)
    ab_surg_pts = np.where(X_input['PREV_AB_SURG_TRR'] == 'Y', 2, 0)
    albuminpts = np.where(X_input['ALBUMIN_TX'] < 2.0, 2, 0)
    Dialypts = np.where(X_input['DIAL_TX'] == 'Y', 3, 0)
    ICUpts = np.where(X_input['MED_COND_TRR'] == 1.0, 6, 0)
    hospts = np.where(X_input['MED_COND_TRR'] == 2.0, 3, 0)
    MELDpts = np.where(X_input['MELD_PELD_LAB_SCORE'] > 30, 4, 0)
    LISUP_pts = np.where(X_input['LIFE_SUP_TRR'] == 'Y', 9, 0)
    #Encepts_pts = np.where(df_liv_Psoft['ENCEPH_TX'] == '2.0' , 2, 0)
    #Encepts_pts = np.where(df_liv_Psoft['ENCEPH_TX'] == '3.0' , 2, 0)
    Encepts_pts = np.where(X_input['ENCEPH_TX'] == 2.0, 2, (np.where(X_input['ENCEPH_TX'] == 3.0, 2, 0)))
    Port_vein_pts = np.where(X_input['PORTAL_VEIN_TRR'] == 'Y', 5, 0)
    ASCITES_pts = np.where(X_input['ASCITES_TX'] == 2.0, 3, (np.where(X_input['ASCITES_TX'] == 3.0, 3, 0)))        

    psoft = agepts + bmipts + tranpts + ab_surg_pts + albuminpts +\
            Dialypts + ICUpts + hospts + MELDpts + LISUP_pts + Encepts_pts + Port_vein_pts + ASCITES_pts
    #dists[(np.where((dists >= r) & (dists <= r + dr)))]
    age_donpts = np.where(((X_input['AGE_DON'] >= 10) & (X_input['AGE_DON'] <= 20)), -2, 0)
    #age_donpts = np.where(np.logical_and(np.greater_equal(df_liv_soft['AGE_DON'],10.0),np.less_equal(df_liv_soft['AGE_DON'],20.0)), -2,0)
    age_donpts = np.where(X_input['AGE_DON'] > 60 ,3,age_donpts)
    cva_donpts = np.where(X_input['COD_CAD_DON'] == 2.0, 2, 0)
    creat_donpts = np.where(X_input['CREAT_DON'] > 1.5, 2, 0)
    alloc_pts  = np.where(X_input['SHARE_TY'] == 3.0, 2, 0)
    cold_ischpts = np.where(((X_input['COLD_ISCH'] >= 0) & (X_input['COLD_ISCH'] <= 6)), -3, 0)
    Soft_score = psoft + age_donpts + cva_donpts + creat_donpts + alloc_pts + cold_ischpts
    X_input['softscore'] = Soft_score
    return X_input

def Barscore(X_input):
    # Bar score rules
    MELD_score_pts = np.where(((X_input['MELD_PELD_LAB_SCORE'] >= 16) & (X_input['MELD_PELD_LAB_SCORE'] <= 25)), 5, 0)
    MELD_score_pts = np.where(((X_input['MELD_PELD_LAB_SCORE'] > 25) & (X_input['MELD_PELD_LAB_SCORE'] <= 35)), 10, MELD_score_pts)
    MELD_score_pts = np.where(X_input['MELD_PELD_LAB_SCORE'] > 35,14, MELD_score_pts)

    PREV_TX_pts = np.where(X_input['PREV_TX'] == 'Y', 4, 0)
    LISUP_pts = np.where(X_input['LIFE_SUP_TRR'] == 'Y', 3, 0)

    age_pts = np.where(((X_input['AGE'] > 40) & (X_input['AGE'] <= 60)), 1, 0)
    age_pts = np.where(X_input['AGE'] > 60,3, age_pts)

    cold_isch_pts = np.where(((X_input['COLD_ISCH'] > 6) & (X_input['COLD_ISCH'] <= 12)), 1, 0)
    cold_isch_pts = np.where(X_input['COLD_ISCH'] > 12,2, cold_isch_pts)

    Don_age_pts = np.where(X_input['AGE_DON'] > 40,1, 0)

    bar_score = MELD_score_pts + PREV_TX_pts + LISUP_pts + age_pts + cold_isch_pts + Don_age_pts
    X_input['barscore'] = bar_score
    return X_input


def get_auc_ci(y_truth, y_pred,num_bootstraps = 1000):
    n_bootstraps = num_bootstraps
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    y_pred=y_pred
    y_true=y_truth
    rng = np.random.RandomState(rng_seed)
    tprs=[]
    aucs=[]
    base_fpr = np.linspace(0, 1, 101)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        fpr, tpr, _ = metrics.roc_curve(y_true[indices],y_pred[indices])
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    mean_auc = metrics.auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tprs + std*2, 1)
    tprs_lower = mean_tprs - std*2
    return base_fpr, mean_tprs, tprs_lower, tprs_upper, mean_auc, std_auc

def plot_auc(truth, reference_model, new_model,n_bootstraps=1000, save=False):
    y_truth = truth
    ref_model = reference_model
    new_model = new_model
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    ref_auc, new_auc = metrics.auc(ref_fpr, ref_tpr), metrics.auc(new_fpr, new_tpr)
    print('ref auc =',ref_auc, '\newn auc = ', new_auc)   
    base_fpr_ref, mean_tprs_ref, tprs_lower_ref, tprs_upper_ref, mean_auc_ref, std_auc_ref=get_auc_ci(y_truth, ref_model,n_bootstraps)
    base_fpr_new, mean_tprs_new, tprs_lower_new, tprs_upper_new, mean_auc_new, std_auc_new=get_auc_ci(y_truth, new_model,n_bootstraps)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(ref_fpr, ref_tpr, color='blue',
             lw=lw, label='Reference raw ROC (AUC = %0.2f)' % ref_auc, linestyle='--')
    plt.plot(base_fpr_ref, mean_tprs_ref, 'b', alpha = 0.8, label=r'Reference mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_ref, (mean_auc_ref-2*std_auc_ref),(mean_auc_ref+2*std_auc_ref)),)
    plt.fill_between(base_fpr_ref, tprs_lower_ref, tprs_upper_ref, color = 'b', alpha = 0.2)
    plt.plot(new_fpr, new_tpr, color='darkorange',
             lw=lw, label='New raw ROC (AUC = %0.2f)' % new_auc, linestyle='--')
    plt.plot(base_fpr_new, mean_tprs_new, 'darkorange', alpha = 0.8, label=r'New mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_new,(mean_auc_new-2*std_auc_new),(mean_auc_new+2*std_auc_new)),)
    plt.fill_between(base_fpr_new, tprs_lower_new, tprs_upper_new, color = 'darkorange', alpha = 0.2)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity', fontsize=18)
    plt.ylabel('Sensitivity', fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')#feat_list = ['mean fractal dimension']
    
