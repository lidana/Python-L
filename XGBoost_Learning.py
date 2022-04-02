import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f = open('cloud.csv')
    df = pd.read_csv(f)
    print(df.head())
    # 0 rows, 1 columns
    df.drop(['SEEDED'],axis=1,inplace=True)
    print(df.head())
    #查看某一列的唯一值，删除重复值
    print(df['binaryClass'].unique())
    #使用replace修改某一列中的某些值
    df['binaryClass'].replace('N',0,regex=True,inplace=True)
    df['binaryClass'].replace('P', 1, regex=True, inplace=True)
    print(df.head())
    # 数据类型
    print(df.dtypes)
    #将某一列转换为数字类型
    df['binaryClass'] = pd.to_numeric(df['binaryClass'])
    print(df.dtypes)
    #数据中标签为N的样本个数，loc函数返回的是满足条件的行
    #注意loc函数使用的是中括号
    print(len(df.loc[df['binaryClass'] == 0]))
    print(len(df.loc[df['binaryClass'] == 1]))
    #将变量为空的地方填上0，前半部分找到满足条件的行，后半部分表示列
    df.loc[(df['SC'] == ' '),'SC'] = 0

    X = df.drop(['binaryClass'],axis=1).copy()
  #  X=np.array(X)
    y = df['binaryClass'].tolist().copy()
   # y=np.array(y)

    #one hot编码所有的object类型的特征
    #pd.get_dummies((X, column = ['SC']))
    #将所有的类别型编码为One hot型，可以将需要处理的特征加在后面
    #X_encoded = pd.get_dummies(X, columns=['SC','TE'])

    #标签为0 或者1，下式可以计算标签为1的数据所占比例
    propo_1 = sum(y) / len(y)
    print(propo_1)
    #保证train和test中的类别比例相似, stratify的作用就是这个
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2,stratify=y)
    print(sum(y_test)/len(y_test))
    print(sum(y_train)/len(y_train))
    print(X_train,y_train,X_test,y_test)
    #训练XGBoost
    my_xgb = xgb.XGBClassifier(objective='binary:logistic',seed=23)
    my_xgb.fit(X_train,y_train, eval_set=[(X_test,y_test)])
    preds = my_xgb.predict(X_test)
    print(preds)
    plot_confusion_matrix(my_xgb,X_test,y_test,values_format='d',display_labels=["yes","no"])
    plt.show()