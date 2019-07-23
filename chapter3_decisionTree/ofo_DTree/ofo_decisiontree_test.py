# coding=utf-8
import pandas as pd
import numpy as np
from datetime import date

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc

# 1、读取数据
## dftest为用户ofo线下优惠券使用预测样本
f1 = open('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/ccf_offline_stage1_test_revised.csv', encoding='utf-8')
dftest = pd.read_csv(f1, keep_default_na=False)
f1.close()
## dfoff为用户线下消费和优惠券领取情况
f2 = open('E:/python/实战/机器学习实战/Cp2_KNN/ofo_knn/ofo_knn/Code/data/ccf_offline_stage1_train.csv', encoding='utf-8')
dfoff = pd.read_csv(f2, keep_default_na=False)
f2.close()
# 查看线下数据的基本情况
print('无优惠券，有消费：' + str(dfoff[(dfoff['Date_received']=='null')&(dfoff['Date']!='null')].shape[0]))
print('无优惠券，无消费：' + str(dfoff[(dfoff['Date_received']=='null')&(dfoff['Date']=='null')].shape[0]))
print('有优惠券，有消费：' + str(dfoff[(dfoff['Date_received']!='null')&(dfoff['Date']!='null')].shape[0]))
print('有优惠券，无消费：' + str(dfoff[(dfoff['Date_received']!='null')&(dfoff['Date']=='null')].shape[0]))

# 2、数据清洗，构造特征
## (1)折扣率变换
print('打折类型有：' + str(dfoff['Discount_rate'].unique()))
def discountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0
def convertRate(row):
    if ':' in row:
        splits = row.split(':')    # 切割完之后都是str格式的
        return 1-float(splits[1])/float(splits[0])
    elif row=='null':
        return 1.0
    else:
        return float(row)
def discountMan(row):
    if ':' in row:
        splits = row.split(':')
        return int(splits[0])
    else:
        return 0
def discountJian(row):
    if ':' in row:
        splits =row.split(':')
        return int(splits[1])
    else:
        return 0
def processData(df):
    df['discount_type'] = df['Discount_rate'].apply(discountType)
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(discountMan)
    df['discount_jian'] = df['Discount_rate'].apply(discountJian)
    return df
dftest = processData(dftest)
dfoff = processData(dfoff)
## 此时创建了4个新的特征：discount_type\discount_rate\discount_man\discount_jian ##

## (2)distance处理
print('dftest距离的类型有：' + str(dftest['Distance'].unique()))
dftest['distance'] = dftest['Distance'].replace('null', -1).astype(int)
dfoff['distance'] = dfoff['Distance'].replace('null', -1).astype(int)
print('dftest距离的类型有：' + str(dftest['distance'].unique()))

## (3)Date_received处理:将日期转化为星期；是否为周末区分出来；将星期几转换为one-hot编码
# 先查看一些日期的范围
date_fw_dfoff = sorted(dfoff[dfoff['Date_received']!='null']['Date_received'])
print('用户线下优惠券领取日期从' + str(date_fw_dfoff[0] + '到' + str(date_fw_dfoff[-1])))
date_xf_dfoff = sorted(dfoff[dfoff['Date']!='null']['Date'])
print('用户线下消费日期从' + str(date_xf_dfoff[0] + '到' + str(date_xf_dfoff[-1])))
# 将日期转换为星期
def getweekday(row):
    if row=='null':
        return 'null'
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getweekday)
dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getweekday)
# 是否为周末区分出来
dftest['weekday_type'] = dftest['weekday'].apply(lambda x:1 if x in [6,7] else 0)
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x:1 if x in[6,7] else 0)
# 将星期转换为one-hot编码
columns = ['weekday_' + str(i) for i in range(1,8)]
dftest[columns] = pd.get_dummies(dftest['weekday'].replace('null',np.nan), columns=columns)
dfoff[columns] = pd.get_dummies(dfoff['weekday'].replace('null',np.nan), columns=columns)
## 所有的特征都已经构造完成，一共有14个 ##

# 3、目标值构造，因为原始数据集中没有给出，所以需要认为构造
### Date_received=='null',标注为-1，不用考虑；
### 15天内领券并消费，为正样本，标注为1；15天外消费及其他情况都标注为0，即为负样本
def getLabels(df):
    if df['Date_received'] == 'null':
        return -1
    if df['Date']!='null':
        diff = pd.to_datetime(df['Date'], format='%Y%m%d') - pd.to_datetime(df['Date_received'], format='%Y%m%d')
        ## 注意：diff的格式为pandas._libs.tslibs.timedeltas.Timedelta，不能与常数直接比较大小
        if diff<pd.Timedelta(15, 'D'):
            return 1
    return 0
dfoff['label'] = dfoff.apply(getLabels, axis=1)
print(dfoff['label'].value_counts())

# 4、建立决策树模型
### 划分训练集和测试集、建立和训练决策树模型、验证（AUC）、测试
# (1)划分训练集和测试集
df = dfoff[dfoff['label']!=-1]
train = df[df['Date_received']<'20160516']
valid = df[(df['Date_received']>='20160516') & (df['Date_received']<='20160615')]
print('训练集的样本数为：' + str(train.shape[0]))
print('验证集的样本数为：' + str(valid.shape[0]))
# (2)建立和训练决策树模型
original_feature = df.columns[-15:-1]    # 参与模型的14个特征
def check_model(data, predictors):
    classifier = DecisionTreeClassifier(random_state=1)
    parameters = {
        'max_leaf_nodes': list(range(2,100)),
        'min_samples_split': [8,10,15]
    }
    # StratifiedKFold与KFold类似，但它是分层采样，确保训练集、测试集中各类别样本的比例与原始数据集中的相同
    folder = StratifiedKFold(n_splits=3, shuffle=True)    # n_splits:Number of folds
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        classifier,
        parameters,
        cv=folder,
        n_jobs=2,
        verbose=1    # Controls the verbosity: the higher, the more messages.
    )
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])
    return grid_search
predictors = original_feature
model = check_model(train, predictors)
# (3)利用valid数据集进行验证
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:,1]
### 计算AUC
vg = valid1.groupby('Coupon_id')
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique())!=2:
        continue
    fpr, tpr, threshold = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))
# (4)测试
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('E:/python/实战/机器学习实战/chapter3_DecisionTree/submit1_DTree.csv', index=False, header=False)
# (5)保存模型
import os, pickle
os.chdir('E:/python/实战/机器学习实战/chapter3_DecisionTree/')
if not os.path.isfile('DTree_model.pkl'):
    with open('DTree_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('DTree_model.pkl', 'rb') as f:
        model = pickle.load(f)


