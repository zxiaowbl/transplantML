# transplantML

Interpretable prediction of mortality in liver transplant recipients based on machine learning

Abstract

Background: Accurate prediction of the mortality of post-liver transplantation is an important but challenging task. It relates to optimizing organ allocation and estimating the risk of possible dysfunction. Existing risk scoring models, such as the Balance of Risk (BAR) score and the Survival Outcomes Following Liver Transplantation (SOFT) score, do not predict the mortality of post-liver transplantation with sufficient accuracy. In this study, we evaluate the performance of machine learning models and establish an explainable machine learning model for predicting mortality in liver transplant
recipients.

Method: The optimal feature set for the prediction of the mortality was selected by a wrapper method based on binary particle swarm optimization (BPSO). With the selected optimal feature set, seven machine learning models were applied to predict mortality over different time windows. The best-performing model was used to predict mortality through a comprehensive comparison and evaluation. An interpretable approach based on machine learning and SHapley Additive exPlanations (SHAP) is used to explicitly explain the model’s decision and make new discoveries.

Results: With regard to predictive power, our results demonstrated that the feature set selected by BPSO outperformed both the feature set in the existing risk score model (BAR score, SOFT score) and the feature set processed by principal component analysis (PCA). The best-performing model, extreme gradient boosting (XGBoost), was found to improve the Area Under a Curve (AUC) values for mortality prediction by 6.7%, 11.6%, and 17.4% at 3 months, 3 years, and 10 years, respectively, compared to the SOFT score. The main predictors of mortality and their impact were discussed for different age groups and different follow-up periods.

Conclusions: Our analysis demonstrates that XGBoost can be an ideal method to assess the mortality risk in liver transplantation. In combination with the SHAP approach, the proposed framework provides a more intuitive and comprehensive interpretation of the predictive model, thereby allowing the clinician to better understand the decision-making process of the model and the impact of factors associated with mortality risk in liver transplantation.

Keywords: Liver transplant, Machine learning, Mortality prediction, Feature selection, Model interpretability
