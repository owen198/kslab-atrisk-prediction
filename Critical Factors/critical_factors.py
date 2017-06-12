import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA


class CriticalFactors:
    def __init__(self):
        datasets_small_name = '1a'
        datasets = pd.read_csv('../data/ncu_data_week_1-6(' + datasets_small_name + ').csv', sep=',')

        features_begin_index = 1
        features_end_index = 21
        total_features = 21

        features_header = list(datasets)[features_begin_index: features_end_index + 1]
        features_val = datasets[features_header].values
        label_header = 'final_score'
        label_val = datasets[label_header].values

        """
        min_max_scaler = preprocessing.MinMaxScaler()
        features_val = min_max_scaler.fit_transform(features_val)
        """

        number_of_comp = 12

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

        features_val = sm.add_constant(features_val)  # sklearn 預設有加入截距，statsmodels沒有，所以要加
        results = sm.OLS(label_val, features_val).fit()
        print results.summary()

        pvalue_l = ['Regression P>|t|']
        for i in range(1, len(results.pvalues)):
            pvalue_l.append(round(results.pvalues[i], 3))

        variable_analysis_results_table.append(pvalue_l)
        variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)
        variable_analysis_results_table.to_csv(
            'result/' + datasets_small_name + '_best_comp_variable_analysis_results_table.csv', index=False)
