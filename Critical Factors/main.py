# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model


datasets_small_name = '1a'
datasets = pd.read_csv('../data/ncu_data_week_1-6(' + datasets_small_name + ').csv', sep=',')










datasets = datasets.loc[(((datasets.video_sum_count - datasets.video_sum_count.mean()) / datasets.video_sum_count.std()).abs() < 1.5)
				& (((datasets.active_sum_count - datasets.active_sum_count.mean()) / datasets.active_sum_count.std()).abs() < 1)
				& (((datasets.final_score - datasets.final_score.mean()) / datasets.final_score.std()) < 0.8)]
print(len(datasets))









features_begin_index = 1
features_end_index = 21
total_features = 21

features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values

"""
min_max_scaler = preprocessing.MinMaxScaler()
features_val = min_max_scaler.fit_transform(features_val)
"""

number_of_comp = 21

pca = PCA(n_components=number_of_comp)
pca.fit(features_val)
features_val = pca.transform(features_val)


variable_analysis_results_table = []
for i in range(total_features):
	row_list = []
	row_list.append(features_header[i])
	for j in pca.components_:
		row_list.append(j[i])
	variable_analysis_results_table.append(row_list)



columns = ['']

for i in range(1, number_of_comp + 1):
	columns.append("comp " + str(i))





features_val = sm.add_constant(features_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
results = sm.OLS(label_val, features_val).fit()
print results.summary()



pvalue_l = ['Regression P>|t|']
for i in range(1, len(results.pvalues)):
	pvalue_l.append(round(results.pvalues[i], 3))

variable_analysis_results_table.append(pvalue_l)
variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)
variable_analysis_results_table.to_csv('result/' + datasets_small_name + '_best_comp_variable_analysis_results_table.csv', index=False)







#從1a,3a 7個comp中有達顯著性的comp定義出，online comp 與 offline comp

#1a online comp = comp 1, 2, 7
#1a offline comp = comp 7
#3a online comp = comp 1, 5
#3a offline comp = comp 5
#跑出ols regression與pMSE, pMAPC指標
#↓

"""

features_val = pd.DataFrame(features_val, columns=columns)


online_comps = pd.DataFrame()
offline_comps = pd.DataFrame()

for i in columns:
	if(i == "comp 1" or i == "comp 2"):
		online_comps[i] = features_val[i]
	#elif(i == "comp 5"):
		#offline_comps[i] = features_val[i]
	elif(i == "comp 7"): # online, offline comp
		online_comps[i] = features_val[i]
		offline_comps[i] = features_val[i]



online_comps_val = online_comps.values
offline_comps_val = offline_comps.values




online_comps_val = sm.add_constant(online_comps_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
online_comps_results = sm.OLS(label_val, online_comps_val).fit()
print("online_comps_results:")
print(online_comps_results.summary())


print("\n------------------------------------------------------------------------------------------------------------\n")

offline_comps_val = sm.add_constant(offline_comps_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
offline_comps_results = sm.OLS(label_val, offline_comps_val).fit()
print("offline_comps_results:")
print(offline_comps_results.summary())









number_of_folds = 10
number_of_cv_evaluation = 100


online_comps_metrics = []
offline_comps_metrics = []

for evaluation_num in range(1, number_of_cv_evaluation + 1):
	kfold = KFold(n_splits=number_of_folds, shuffle=True)
	kfold_split_num = 1
	for train_index, test_index in kfold.split(online_comps_val):
		online_comps_val_train, online_comps_val_test = online_comps_val[train_index], online_comps_val[test_index]
		offline_comps_val_train, offline_comps_val_test = offline_comps_val[train_index], offline_comps_val[test_index]
		label_val_train, label_val_test = label_val[train_index], label_val[test_index]
		
		
		#min_max_scaler = preprocessing.MinMaxScaler()
		#min_max_scaler.fit(online_comps_val_train)
		
		#online_comps_val_train = min_max_scaler.transform(online_comps_val_train)
		#online_comps_val_test = min_max_scaler.transform(online_comps_val_test)
		

		MLR = linear_model.LinearRegression()
		MLR.fit(online_comps_val_train, label_val_train)
		label_val_predict = MLR.predict(online_comps_val_test)


		#處理預測值超過不合理範圍
		for i in range(len(label_val_predict)):
			if label_val_predict[i] > 104.16:
				label_val_predict[i] = 104.16
			elif label_val_predict[i] < 0:
				label_val_predict[i] = 0.0
		#處理預測值超過不合理範圍
		
		
		#pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / label_val_test))
		pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
		pMSE = np.mean((label_val_predict - label_val_test) ** 2)
		online_comps_metrics.append([evaluation_num, kfold_split_num, pMAPC, pMSE])



		MLR = linear_model.LinearRegression()
		MLR.fit(offline_comps_val_train, label_val_train)
		label_val_predict = MLR.predict(offline_comps_val_test)


		#處理預測值超過不合理範圍
		for i in range(len(label_val_predict)):
			if label_val_predict[i] > 104.16:
				label_val_predict[i] = 104.16
			elif label_val_predict[i] < 0:
				label_val_predict[i] = 0.0
		#處理預測值超過不合理範圍
		
		
		#pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / label_val_test))
		pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
		pMSE = np.mean((label_val_predict - label_val_test) ** 2)
		offline_comps_metrics.append([evaluation_num, kfold_split_num, pMAPC, pMSE])
		kfold_split_num = kfold_split_num + 1




print("--------------------------------------------------------------------------------------")
online_comps_metrics = pd.DataFrame(online_comps_metrics, columns=['evaluation_num', 'kfold_split_num', 'pMAPC', 'pMSE'])
online_comps_metrics = online_comps_metrics.mean()
print("online_comps Mean pMSE:" + str(online_comps_metrics['pMSE']))
print("online_comps Mean pMAPC:" + str(online_comps_metrics['pMAPC']))

offline_comps_metrics = pd.DataFrame(offline_comps_metrics, columns=['evaluation_num', 'kfold_split_num', 'pMAPC', 'pMSE'])
offline_comps_metrics = offline_comps_metrics.mean()
print("offline_comps Mean pMSE:" + str(offline_comps_metrics['pMSE']))
print("offline_comps Mean pMAPC:" + str(offline_comps_metrics['pMAPC']))
"""