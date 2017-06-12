# coding: utf-8
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA


class CriticalFactors:
    _LABEL_HEADER = 'final_score'
    _NUMBER_OF_COMPONENTS = 12

    def __init__(self, data, features_begin_index, features_end_index):
        self.features_begin_index = features_begin_index
        self.features_end_index = features_end_index
        self.total_features = features_end_index - features_begin_index + 1
        self.data = data
        self._process_data()

    def _process_data(self):

        # datasets_small_name = '1a'
        # datasets = pd.read_csv('../data/ncu_data_week_1-6(' + datasets_small_name + ').csv', sep=',')
        #
        # features_begin_index = 1
        # features_end_index = 21

        features_header = list(self.data)[self.features_begin_index: self.features_end_index + 1]
        features_val = self.data[features_header].values
        label_header = self._LABEL_HEADER
        label_val = self.data[label_header].values

        """
        min_max_scaler = preprocessing.MinMaxScaler()
        features_val = min_max_scaler.fit_transform(features_val)
        """

        pca = PCA(n_components=self._NUMBER_OF_COMPONENTS)
        pca.fit(features_val)
        features_val = pca.transform(features_val)

        variable_analysis_results_table = []
        for i in range(self.total_features):
            row_list = []
            row_list.append(features_header[i])
            for j in pca.components_:
                row_list.append(j[i])
            variable_analysis_results_table.append(row_list)

        columns = ['']

        for i in range(1, self._NUMBER_OF_COMPONENTS + 1):
            columns.append("comp " + str(i))

        features_val = sm.add_constant(features_val)  # sklearn 預設有加入截距，statsmodels沒有，所以要加
        results = sm.OLS(label_val, features_val).fit()
        print results.summary()

        pvalue_l = ['Regression P>|t|']
        for i in range(1, len(results.pvalues)):
            pvalue_l.append(round(results.pvalues[i], 3))

        variable_analysis_results_table.append(pvalue_l)
        self.variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)

    def save_to_csv(self, path):
        self.variable_analysis_results_table.to_csv(path, index=False)
