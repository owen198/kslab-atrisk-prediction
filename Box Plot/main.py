# coding: utf-8

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import wilcoxon
from func import *




all_datasets_metrics = []
online_offline_all_datasets_metrics = []
datasets_name = ['ncu_data_week_1-6(1a)', 'ncu_data_week_1-12(2a)', 'ncu_data_week_1-18(3a)', 'ncu_data_week_7-12(2d)', 'ncu_data_week_13-18(3d)']
datasets_small_name = ['1a', '2a', '3a', '2d', '3d']
datasets_index = 0

for dataset_name in datasets_name:
	datasets = pd.read_csv('../data/' + dataset_name + '.csv', sep=',')



	online_features_begin_index = 1
	online_features_end_index = 21

	features_header = list(datasets)[online_features_begin_index : online_features_end_index + 1]
	online_features_val = datasets[features_header].values


	label_header = 'final_score'
	label_val = datasets[label_header].values

	number_of_folds = 10
	total_features = 21



	number_of_cv_evaluation = 20

	metrics_list = []
	online_metrics_list = []
	offline_metrics_list = []

	for evaluation_num in range(1, number_of_cv_evaluation + 1):
		kfold = KFold(n_splits=number_of_folds, shuffle=True)
		kfold_split_num = 1

		for train_index, test_index in kfold.split(online_features_val):
			online_features_val_train, online_features_val_test = online_features_val[train_index], online_features_val[test_index]
			label_val_train, label_val_test = label_val[train_index], label_val[test_index]

			

			for number_of_comp in range(1, total_features + 1):
				pca = PCA(n_components=number_of_comp)
				pca.fit(online_features_val_train)
				online_features_pca_val_train = pca.transform(online_features_val_train)
				MLR = linear_model.LinearRegression()
				MLR.fit(online_features_pca_val_train, label_val_train)
				online_features_pca_val_test = pca.transform(online_features_val_test)
				label_val_predict_online = MLR.predict(online_features_pca_val_test)
				#處理預測值超過不合理範圍
				for i in range(len(label_val_predict_online)):
					if label_val_predict_online[i] > 104.16:
						label_val_predict_online[i] = 104.16
					elif label_val_predict_online[i] < 0:
						label_val_predict_online[i] = 0.0
				#處理預測值超過不合理範圍


				pMAPC = 1 - np.mean(abs((label_val_predict_online - label_val_test) / label_val_test))
				#pMAPC = 1 - np.mean(abs((label_val_predict_online - label_val_test) / np.mean(label_val)))
				pMSE = np.mean((label_val_predict_online - label_val_test) ** 2)
				metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, pMAPC, pMSE])


			kfold_split_num = kfold_split_num + 1

	metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])

	metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
	metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
	metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)


	all_datasets_metrics.append(metrics_dataframe)



	datasets_index = datasets_index + 1





wilcoxon_pMAPC_table = []
wilcoxon_pMSE_table = []
all_comp_predictive_metrics_table = []
row = 0
for i in all_datasets_metrics:
	wilcoxon_pMAPC_row = [datasets_small_name[row]]
	wilcoxon_pMSE_row = [datasets_small_name[row]]
	mean_pMAPC = np.mean(i['pMAPC'].values)
	mean_pMSE = np.mean(i['pMSE'].values)
	all_comp_predictive_metrics_row = [datasets_small_name[row], mean_pMAPC, mean_pMSE]
	all_comp_predictive_metrics_table.append(all_comp_predictive_metrics_row)

	for j in all_datasets_metrics:
		z_statistic, p_value = wilcoxon(i['pMAPC'].values - j['pMAPC'].values)
		wilcoxon_pMAPC_row.append(p_value)
		z_statistic, p_value = wilcoxon(i['pMSE'].values - j['pMSE'].values)
		wilcoxon_pMSE_row.append(p_value)
	wilcoxon_pMAPC_table.append(wilcoxon_pMAPC_row)
	wilcoxon_pMSE_table.append(wilcoxon_pMSE_row)
	
	row = row + 1




columns = []
columns.append('')
for name in datasets_small_name:
	columns.append(name)
wilcoxon_pMAPC_table = pd.DataFrame(wilcoxon_pMAPC_table, columns=columns)
wilcoxon_pMSE_table = pd.DataFrame(wilcoxon_pMSE_table, columns=columns)
all_comp_predictive_metrics_table = pd.DataFrame(all_comp_predictive_metrics_table, columns=['', 'pMAPC', 'pMSE'])



pMSE_and_pMAPC_Table = all_comp_predictive_metrics_table.merge(wilcoxon_pMAPC_table, left_on=[""], right_on=[""], how='inner')
pMSE_and_pMAPC_Table = pMSE_and_pMAPC_Table.merge(wilcoxon_pMSE_table, left_on=[""], right_on=[""], how='inner')



pMSE_and_pMAPC_Table.columns = ['', 'pMAPC', 'pMSE', '1a(pMAPC)', '2a(pMAPC)', '3a(pMAPC)', '2d(pMAPC)', '3d(pMAPC)', '1a(pMSE)', '2a(pMSE)', '3a(pMSE)', '2d(pMSE)', '3d(pMSE)']
pMSE_and_pMAPC_Table.to_csv('result/pMSE_and_pMAPC_Table.csv', index=False)





#boxplot of pMSE and pMAPC
all_datasets_pMSE = []
all_datasets_pMPAC = []
for i in all_datasets_metrics:
	all_datasets_pMSE.append(i['pMSE'].values)
	all_datasets_pMPAC.append(i['pMAPC'].values)



generate_boxplot(all_datasets_pMSE, 'pMSE Comparison between different datasets', datasets_small_name)
generate_boxplot(all_datasets_pMPAC, 'pMAPC Comparison between different datasets', datasets_small_name)
#boxplot of pMSE and pMAPC



index = 0
for i in all_datasets_metrics:
	i.to_csv('result/PCR_' + datasets_small_name[index] + '.csv', index=False)
	index = index + 1