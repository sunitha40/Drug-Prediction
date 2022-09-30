import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle



dataset=pd.read_csv("C:\\Users\\SRamayanam\\Downloads\\drug200.csv")



dataset['Sex']=dataset['Sex'].map({'F':0,'M':1})
dataset['BP']=dataset['BP'].map({'LOW':0,'NORMAL':1,'HIGH':2})
dataset['Cholesterol']=dataset['Cholesterol'].map({'HIGH':2,'NORMAL':1})



outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data:
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_iqr(dataset['Na_to_K'])




median = np.median(dataset['Na_to_K'])# Replace with median
for i in sample_outliers:
    dataset['Na_to_K'] = np.where(dataset['Na_to_K']==i, median, dataset['Na_to_K'])



X=dataset.drop(['Drug'],axis=1)
y=dataset['Drug']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)




from sklearn.tree import DecisionTreeClassifier
gini=DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=0)
gini.fit(X_train,y_train)



pickle.dump(gini, open('model.pkl','wb'))



model=pickle.load(open('model.pkl','rb'))

