

```python
!pip3 install scipy
!pip3 install xgboost
```

    Requirement already satisfied (use --upgrade to upgrade): scipy in /usr/local/lib/python3.5/dist-packages
    Requirement already satisfied (use --upgrade to upgrade): numpy>=1.8.2 in /usr/local/lib/python3.5/dist-packages (from scipy)
    [33mYou are using pip version 8.1.1, however version 18.0 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Collecting xgboost
      Downloading https://files.pythonhosted.org/packages/06/7a/442f7da21792566012e5c7e5a7dffa44c1b6cc05c0c27856bbc8a7718b28/xgboost-0.72.1-py2.py3-none-manylinux1_x86_64.whl (18.4MB)
    [K    100% |████████████████████████████████| 18.4MB 59kB/s eta 0:00:01    16% |█████▏                          | 3.0MB 3.4MB/s eta 0:00:05    21% |███████                         | 4.0MB 3.6MB/s eta 0:00:05    37% |████████████▏                   | 7.0MB 5.2MB/s eta 0:00:03    51% |████████████████▌               | 9.5MB 4.7MB/s eta 0:00:02
    [?25hRequirement already satisfied (use --upgrade to upgrade): scipy in /usr/local/lib/python3.5/dist-packages (from xgboost)
    Requirement already satisfied (use --upgrade to upgrade): numpy in /usr/local/lib/python3.5/dist-packages (from xgboost)
    Installing collected packages: xgboost
    Successfully installed xgboost-0.72.1
    [33mYou are using pip version 8.1.1, however version 18.0 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
import pandas as pd
```


```python
kyoto_x_df_1 = pd.read_csv('dataset1_features.csv')
kyoto_x_df_2 = pd.read_csv('dataset2_features.csv')
```


```python
kyoto_y_df_1 = pd.read_csv('data1_score.csv')
kyoto_y_df_2 = pd.read_csv('data2_score.csv')
```


```python
kyoto_y_df_2['class'] = kyoto_y_df_2['score']>80
kyoto_y_df_1['class'] = kyoto_y_df_1['score']>80
```


```python
kyoto_df_1 = pd.merge(kyoto_x_df_1, kyoto_y_df_1, on='userid')
kyoto_df_1 = kyoto_df_1.drop(['userid', 'score'], axis=1)
kyoto_df_1.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookmarkc</th>
      <th>closec</th>
      <th>markerc</th>
      <th>memoc</th>
      <th>mobilec</th>
      <th>openc</th>
      <th>pcc</th>
      <th>tabletc</th>
      <th>watchc</th>
      <th>SEARCH</th>
      <th>...</th>
      <th>Page_JumpC</th>
      <th>Add_BookmarkC</th>
      <th>Delete_BookmarkC</th>
      <th>Add_MemoC</th>
      <th>Delete_MemoC</th>
      <th>Change_MemoC</th>
      <th>Add_MarkerC</th>
      <th>Delete_MarkerC</th>
      <th>Readpages</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>1240</td>
      <td>66</td>
      <td>1306</td>
      <td>0</td>
      <td>...</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1211</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>45</td>
      <td>5</td>
      <td>409</td>
      <td>0</td>
      <td>454</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>443</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>89</td>
      <td>0</td>
      <td>89</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>77</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>540</td>
      <td>0</td>
      <td>540</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>530</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
kyoto_df_2 = pd.merge(kyoto_x_df_2, kyoto_y_df_2, on='userid')
kyoto_df_2 = kyoto_df_2.drop(['userid', 'score'], axis=1)
kyoto_df_2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookmarkc</th>
      <th>closec</th>
      <th>markerc</th>
      <th>memoc</th>
      <th>mobilec</th>
      <th>openc</th>
      <th>pcc</th>
      <th>tabletc</th>
      <th>watchc</th>
      <th>SEARCH</th>
      <th>...</th>
      <th>Page_JumpC</th>
      <th>Add_BookmarkC</th>
      <th>Delete_BookmarkC</th>
      <th>Add_MemoC</th>
      <th>Delete_MemoC</th>
      <th>Change_MemoC</th>
      <th>Add_MarkerC</th>
      <th>Delete_MarkerC</th>
      <th>Readpages</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>371</td>
      <td>0</td>
      <td>371</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>361</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>731</td>
      <td>0</td>
      <td>731</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>699</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>23</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>231</td>
      <td>0</td>
      <td>231</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>174</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>122</td>
      <td>0</td>
      <td>122</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>103</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1068</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1053</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split  

features = -3

x_train = kyoto_df_1[list(kyoto_df_1)[0:features]]
y_train = kyoto_df_1[list(kyoto_df_1)[-1]]

x_test = kyoto_df_2[list(kyoto_df_2)[0:features]]
y_test = kyoto_df_2[list(kyoto_df_2)[-1]]

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) 
```


```python
from sklearn.svm import SVC  

svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)  
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
from sklearn.metrics import accuracy_score

y_pred = svclassifier.predict(x_test) 
acc_ext = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)

print (acc_ext, auc)
```

    0.6 0.5584239130434783



```python
y_pred = svclassifier.predict(x_train) 
acc_int = accuracy_score(y_train, y_pred)
acc_int
```




    0.7169811320754716




```python
err = 1 - acc_ext
PAD = 2 * (1 - 2 * err)
PAD
```




    0.3999999999999999




```python
err = 1 - acc_int
PAD = 2 * (1 - 2 * err)
PAD
```




    0.8679245283018866



# Xgboost


```python
import datetime
import time
import pandas as pd
#from influxdb import DataFrameClient
from sklearn.externals import joblib
import numpy as np
import collections
from xgboost.sklearn import XGBClassifier
from pathlib import Path
from sklearn import metrics
```


```python
xgb1 = XGBClassifier()

xgb1.fit(x_train, y_train, eval_metric='auc')

#display results
y_pred = xgb1.predict(x_test)

acc_xgb = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)

print (acc_xgb, auc)

```

    0.6727272727272727 0.6331521739130435


    /usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:

