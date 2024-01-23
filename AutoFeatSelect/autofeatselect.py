#pip install autofeatselect
#pip install sklearn
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from autofeatselect import CorrelationCalculator, FeatureSelector, AutoFeatureSelect
zf_test = ZipFile('C:/Users/namit/Documents/Sumedha/ML/Safe Driver Prediction Kaggle/test.csv.zip', 'r')
df_test = pd.read_csv(zf_test.open('test.csv'))
zf_test.close()
zf_train = ZipFile('C:/Users/namit/Documents/Sumedha/ML/Safe Driver Prediction Kaggle/train.csv.zip', 'r')
df_train = pd.read_csv(zf_train.open('train.csv'))
zf_train.close()
print(df_test)
print(df_train)
df_train.drop('id', axis=1, inplace = True)
response = 'target'
cat_feats = [c for c in df_train.columns if '_cat' in c]
bin_feats = [c for c in df_train.columns if '_bin' in c]
cat_feats = cat_feats+bin_feats
num_feats = [c for c in df_train.columns if c not in cat_feats+[response]]
df_train[num_feats] = df_train[num_feats].astype('float')
df_train[cat_feats] = df_train[cat_feats].astype('object')
df_train.replace(-1, np.nan, inplace=True)
print(df_train['ps_ind_15'])
print(num_feats)
print(df_train[response])
X_train, X_test, y_train, y_test = train_test_split(df_train[num_feats+cat_feats],
                                                    df_train[response],
                                                    test_size=0.2,
                                                    random_state=42)
