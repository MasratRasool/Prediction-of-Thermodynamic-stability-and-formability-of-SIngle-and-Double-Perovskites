#!/usr/bin/env python
# coding: utf-8

# ### Import Necessary Libraries

# In[ ]:


#Creation of Numpy Arrays,Scientific computing and linear Algebra.  
import numpy as np     

#For Loading DataSet,Data Manipulation,Data cleaning and Stastics Analysis.
import pandas as pd

#For Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Dataset

# In[ ]:


data=pd.read_csv("formability_database1.csv")


# In[ ]:


data.head()


# ### Shape of the Dataset.

# In[ ]:


data.shape


# ### Displaying the Column Names

# In[ ]:


data.columns


# ### Finding Stastical Summary

# In[ ]:


data.describe().T


# ### Finding Duplicates rows.

# In[ ]:


data.duplicated().any()


# There is no duplicate rows in the dataset

# ### Finding Column Names,Non-Null values and Datatypes.

# In[ ]:


data.info()


# ### Finding Nullvalues.

# In[ ]:


data.isnull().any()


# ### Finding data types

# In[ ]:


data.dtypes


# ### Describing Dataset with object Datatype.

# In[ ]:


data.select_dtypes(include=['object'])


# ### Converting Object Datatype to Numerical Datatype.

# In[ ]:


# convert A column is object type to category type
data["A"] = data["A"].astype('category')
data["A"] = data["A"].cat.codes


# In[ ]:


# convert A' column is object type to category type
data["A'"] = data["A'"].astype('category')
data["A'"] = data["A'"].cat.codes


# In[ ]:


# convert B column is object type to category type
data["B"] = data["B"].astype('category')
data["B"] = data["B"].cat.codes


# In[ ]:


# convert B' column is object type to category type
data["B'"] = data["B'"].astype('category')
data["B'"] = data["B'"].cat.codes


# In[ ]:


# convert X1 column is object type to category type
data["X1"] = data["X1"].astype('category')
data["X1"] = data["X1"].cat.codes


# In[ ]:


# convert type column is object type to category type
data["type"] = data["type"].astype('category')
data["type"] = data["type"].cat.codes


# In[ ]:


# convert functional group column is object type to category type
data["functional group"] = data["functional group"].astype('category')
data["functional group"] = data["functional group"].cat.codes


# ### After Converting Checking the datatypes.

# In[ ]:


data.dtypes


# In[ ]:


data.head()
data_per=data


# ### Spittng the dataset and choosing the Target Column

# In[ ]:


X = data.iloc[:,0:40]  #independent columns
y = data.iloc[:,-1]    #target column i.e pervoskite


# ### Heatmap For Correlation

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(data= data.corr(), annot=True, cmap='viridis')        


# ### Correlation with independent variable

# In[ ]:


X.corrwith(y).plot.bar(figsize = (15, 10), title = "Correlation with Target", fontsize = 10,grid = True)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# ### Splitting Data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=12)


# ### ExtraTrees Classifier

# In[ ]:


et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)


# In[ ]:


et.feature_importances_


# ### Calculating the importance of each feature using ET

# In[ ]:


plt.barh(X.columns, et.feature_importances_)


# In[ ]:


sorted_idx = et.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], et.feature_importances_[sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Importance")


# In[ ]:


X.columns[sorted_idx]


# ### Top 20 Columns 

# In[ ]:


et_df1=data[['B_X-','A_e_affin+', 'A_HOMO+', 'functional group', 'A_IE+', 'B_OS',
       "B'", "A_LUMO-", "B_HOMO+", "B_e_affin+", 'B', "B_IE+", "A_X+","B_HOMO-",
            'A_Z_radii+', 'B_LUMO+', 'B_X+', 'B_Z_radii+', 'μ', 't']]


# In[ ]:


et_df1.shape


# In[ ]:


et_df1


# In[ ]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = et_df1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='viridis')


# ### Splitting data train and test

# In[ ]:


X = et_df1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
df1=X


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve


# ### RandomForestClassifier

# In[485]:


clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# ### Creating a confusion matrix

# In[486]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ### classification_report

# In[487]:


print(classification_report(y_test, y_pred))


# ### roc_auc_score

# In[488]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[489]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Precision-Recall Curve

# In[490]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### ExtraTreesClassifier

# In[491]:


clf = ExtraTreesClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
et=metrics.accuracy_score(y_test, y_pred)
et
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[492]:


et=metrics.accuracy_score(y_test, y_pred)
et


# In[493]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ### classification_report

# In[494]:


print(classification_report(y_test, y_pred))


# ### roc_auc_score

# In[495]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[496]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Precision-Recall Curve

# In[497]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### Gradient boosting

# In[498]:


clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
gb=metrics.accuracy_score(y_test, y_pred)
gb
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[499]:


gb=metrics.accuracy_score(y_test, y_pred)
gb


# In[500]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ### classification_report

# In[501]:


print(classification_report(y_test, y_pred))


# ### roc_auc_score

# In[502]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for GradientBoostingClassifier: ', roc_auc_score(y_test, y_pred))


# In[503]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - GradientBoostingClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Precision-Recall Curve

# In[504]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### SVC

# In[505]:


svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
sv=metrics.accuracy_score(y_test, y_pred)
sv
rfc_cv_score = cross_val_score(svc, X, y, cv=5)
rfc_cv_score


# In[506]:


sv=metrics.accuracy_score(y_test, y_pred)
sv


# In[ ]:





# In[507]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ### classification_report

# In[508]:


print(classification_report(y_test, y_pred))


# ### roc_auc_score

# In[509]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[510]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Precision-Recall Curve

# In[511]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### Gaussian NB

# In[512]:


gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred=svc.predict(X_test)
nb=metrics.accuracy_score(y_test, y_pred)
nb
rfc_cv_score = cross_val_score(gnb, X, y, cv=5)
rfc_cv_score


# In[513]:


nb=metrics.accuracy_score(y_test, y_pred)
nb


# In[ ]:





# In[514]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ### classification_report

# In[515]:


print(classification_report(y_test, y_pred))


# ### roc_auc_score

# In[516]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[517]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Precision-Recall Curve

# In[518]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[519]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[520]:


X=et_df
X
y


# In[521]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape


# In[522]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE


# In[523]:


sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=20))
sel.fit(X_train, y_train)


# In[524]:


sel.get_support()


# In[525]:


features = X_train.columns[sel.get_support()]


# In[526]:


X_train


# In[527]:


X_train_rfe = sel.transform(X_train)
X_test_rfe = sel.transform(X_test)


# In[528]:


def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    #f.append(acc)
    


# In[529]:


get_ipython().run_cell_magic('time', '', 'run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)')


# In[530]:


get_ipython().run_cell_magic('time', '', 'run_randomForest(X_train, X_test, y_train, y_test)')


# In[531]:


y_train.shape


# ### RandomForestClassifier

# In[532]:


r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    l.append(index)
    r.append(key)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)

            


# In[533]:


randf = pd.DataFrame(list(zip(l, f)),columns =['features', 'r_accuracy'])
randf


# In[534]:


randf.plot(kind = 'line',
        x = 'features',
        y = 'r_accuracy',
        color = 'green')


# ### GradientBoostingClassifier

# In[535]:


from sklearn.ensemble import GradientBoostingClassifier
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
   
    f.append(acc)


# In[536]:


gb = pd.DataFrame(list(zip(l, f)),columns =['features', 'g_accuracy'])
gb


# In[537]:


gb.plot(kind = 'line',
        x = 'features',
        y = 'g_accuracy',
        color = 'blue')


# ### ExtraTreesClassifier

# In[538]:


r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)


# In[539]:


ext = pd.DataFrame(list(zip(l, f)),columns =['features', 'e_accuracy'])
ext


# In[540]:


ext.plot(kind = 'line',
        x = 'features',
        y = 'e_accuracy',
        color = 'orange')


# ### SVC

# In[541]:


from sklearn.svm import LinearSVC
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    print()
    f.append(acc)


# In[542]:


svc = pd.DataFrame(list(zip(l, f)),columns =['features', 'svc_accuracy'])
svc


# In[543]:


svc.plot(kind = 'line',
        x = 'features',
        y = 'svc_accuracy',
        color = 'black')


# ### GaussianNB

# In[544]:


from sklearn.naive_bayes import GaussianNB
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = GaussianNB()
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)


# In[545]:


nb = pd.DataFrame(list(zip(l, f)),columns =['features', 'nb_accuracy'])
nb


# In[546]:


nb.plot(kind = 'line',
        x = 'features',
        y = 'nb_accuracy',
        color = 'yellow')


# In[547]:


all_alg=randf


# In[548]:


all_alg['e_accuracy']=ext['e_accuracy']
all_alg['svc_accuracy']=svc['svc_accuracy']
all_alg['nb_accuracy']=nb['nb_accuracy']
all_alg['g_accuracy']=gb['g_accuracy']


# In[549]:


all_alg.plot(kind = 'line',
        x = 'features',
        y = ['nb_accuracy','e_accuracy','svc_accuracy','g_accuracy'])


# In[ ]:





# # generating new features 

# In[550]:


et_df.columns


# In[551]:


et_df["BX"] = et_df["B_X+"] + et_df["B_X-"]


# In[ ]:


et_df["BX"] = et_df["B_X+"] + et_df["B_X-"]
et_df["HOMO"]=et_df["A_HOMO+"] + et_df["B_HOMO+"]
et_df["LUMO"] = et_df["B_LUMO+"] + et_df["A_LUMO+"]
et_df["radii"] = et_df["A_Z_radii+"] + et_df["B_Z_radii+"]


# In[552]:


et_df.head()


# In[553]:


et_df.columns


# In[554]:


final=et_df.drop(['B_X-','B_X+'], axis = 1)


# In[555]:


final.shape


# In[ ]:





# In[556]:


x=final
x.head()


# In[557]:


y.head()


# In[558]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


# ### RandomForestClassifier

# In[559]:


clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[560]:


# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[561]:


print(classification_report(y_test, y_pred))


# In[562]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[563]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[564]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### ExtraTreesClassifier

# In[565]:


clf = ExtraTreesClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
et=metrics.accuracy_score(y_test, y_pred)
et
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[566]:


et=metrics.accuracy_score(y_test, y_pred)
et


# In[567]:


# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[568]:


print(classification_report(y_test, y_pred))


# In[569]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[570]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[571]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### GradientBoostingClassifier

# In[572]:


clf=GradientBoostingClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
gb=metrics.accuracy_score(y_test, y_pred)
gb
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[573]:


gb=metrics.accuracy_score(y_test, y_pred)
gb


# In[574]:


# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[575]:


print(classification_report(y_test, y_pred))


# In[576]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[577]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[578]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### svc

# In[579]:


svc=SVC() #Default hyperparameters
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
sv=metrics.accuracy_score(y_test, y_pred)
sv
rfc_cv_score = cross_val_score(svc, X, y, cv=5)
rfc_cv_score


# In[580]:


sv=metrics.accuracy_score(y_test, y_pred)
sv


# In[581]:


# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[582]:


print(classification_report(y_test, y_pred))


# In[583]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[584]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[585]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### GaussianNB

# In[586]:


gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred=svc.predict(x_test)
nb=metrics.accuracy_score(y_test, y_pred)
nb
rfc_cv_score = cross_val_score(gnb, X, y, cv=5)
rfc_cv_score


# In[587]:


nb=metrics.accuracy_score(y_test, y_pred)
nb


# In[588]:


# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[589]:


print(classification_report(y_test, y_pred))


# In[590]:


FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))


# In[591]:


plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[592]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[593]:


# Importing standardscalar module
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

# fitting
scalar.fit(final)
scaled_data = scalar.transform(x)

# Importing PCA
from sklearn.decomposition import PCA

# Let's say, components = 2
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

x_pca.shape


# In[594]:


# giving a larger plot
plt.figure(figsize =(8, 6))

plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y, cmap ='plasma')

# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[595]:


# components
pca.components_


# In[596]:


df_comp = pd.DataFrame(pca.components_, columns = x.columns)

plt.figure(figsize =(14, 6))

# plotting heatmap
sns.heatmap(df_comp)


# In[597]:


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel ='rbf', fit_inverse_transform=True, gamma=0.5,n_components=2)
X_kpca = kpca.fit_transform(x)
  
plt.title("Kernel PCA")
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c = y)
plt.show()


# # loading stability dataset

# In[835]:


data = pd.read_csv("stability_databasemev2.csv")
data
df2=data
df2


# In[836]:


#Creation of Numpy Arrays,Scientific computing and linear Algebra.  
import numpy as np     

#For Loading DataSet,Data Manipulation,Data cleaning and Stastics Analysis.
import pandas as pd

#For Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[837]:


data.head()


# In[838]:


data.columns


# In[808]:


### Finding Stastical Summary


# In[839]:


data.describe().T


# In[810]:


data.info()


# In[840]:


data.isnull().any()


# In[812]:


data.dtypes


# In[841]:


data.select_dtypes(include=['object'])


# ### Converting Object Datatype to Numerical Datatype.

# In[842]:


# convert A column is object type to category type
data["A"] = data["A"].astype('category')
data["A"] = data["A"].cat.codes


# In[843]:


# convert A' column is object type to category type
data["A'"] = data["A'"].astype('category')
data["A'"] = data["A'"].cat.codes


# In[844]:


# convert B column is object type to category type
data["B"] = data["B"].astype('category')
data["B"] = data["B"].cat.codes


# In[845]:


# convert B' column is object type to category type
data["B'"] = data["B'"].astype('category')
data["B'"] = data["B'"].cat.codes


# In[846]:


# convert X1 column is object type to category type
data["X1"] = data["X1"].astype('category')
data["X1"] = data["X1"].cat.codes


# In[847]:


# convert type column is object type to category type
data["type"] = data["type"].astype('category')
data["type"] = data["type"].cat.codes


# In[848]:


# convert functional group column is object type to category type
data["functional group"] = data["functional group"].astype('category')
data["functional group"] = data["functional group"].cat.codes


# # After conversion: checking the data types

# In[849]:


data.dtypes


# In[850]:


data.head()


# In[851]:


data.rename(columns={"μ𝐵 ̅	":"m"},inplace=True)


# In[852]:


data['AOS'] = data['A_OS']+data["A'_OS"]
data['AA'] = data['A']+data["A'"]
data['BB'] = data['B']+data["B'"]
data['A_HOMO'] = data['A_HOMO-']+data["A_HOMO+"]
data['A_IE'] = data['A_IE-']+data["A_IE+"]
data['A_LUMO'] = data['A_LUMO-']+data["A_LUMO+"]
data['A_X'] = data['A_X-']+data["A_X+"]
data['A_Z_radii'] = data['A_Z_radii-']+data["A_Z_radii+"]
data['A_e_affin'] = data['A_e_affin-']+data["A_e_affin+"]
data['BOS'] = data['B_OS']+data["B'_OS"]
data['B_HOMO'] = data['B_HOMO-']+data["B_HOMO+"]
data['B_IE'] = data['B_IE-']+data["B_IE+"]
data['B_LUMO'] = data['B_LUMO-']+data["B_LUMO+"]
data['B_X'] = data['B_X-']+data["B_X+"]
data['B_Z_radii'] = data['B_Z_radii-']+data["B_Z_radii+"]
data['B_e_affin'] = data['B_e_affin-']+data["B_e_affin+"]


# In[853]:


data.columns


# In[854]:


data = data[['functional group','AOS', 'AA', 'A_HOMO', 'A_IE', 'A_LUMO', 'A_X', 'A_Z_radii',
       'A_e_affin', 'BOS', 'B_HOMO', 'B_IE', 'B_LUMO', 'B_X', 'B_Z_radii',
       'B_e_affin','X1','X1_OS', 'e_above_hull(meV)', 'μ', 'μĀ','stable', 't', 'type']]


# In[865]:


data
dfs=data
dfs


# In[866]:


data.shape


# # Spittng the dataset and choosing the Target Column

# In[867]:


X = data.drop(["stable"],axis=1) 
y = data['stable']    


# # Heatmap for correlation

# In[868]:


plt.figure(figsize=(15,10))
sns.heatmap(data= data.corr(), annot=True, cmap='viridis')     


# In[869]:


X.corrwith(y).plot.bar(figsize = (15, 10), title = "Correlation with Target", fontsize = 10,grid = True)


# In[870]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# # Splitting data

# In[871]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=12)


# In[872]:


### ExtraTrees Classifier


# In[873]:


et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)


# In[874]:


et.feature_importances_


# In[875]:


plt.barh(X.columns, et.feature_importances_)


# In[876]:


sorted_idx = et.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], et.feature_importances_[sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Importance")


# In[877]:


X.columns[sorted_idx]


# # Top 20 columns

# In[880]:


et_df2=data[['e_above_hull(meV)',"B_HOMO", "A_IE", "BOS", "B_LUMO", "functional group", "AOS","A_X", "B_e_affin", 'B_X', "A_e_affin", "μ", "type", "t",
            "B_Z_radii", "B_IE", "B_LUMO", "B_HOMO"]]


# In[881]:


et_df2.head()


# In[882]:


dfs=et_df2
dfs


# In[883]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = et_df2.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='viridis')


# # splitting dataset

# In[884]:


X = et_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))


# In[885]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[886]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve


# In[887]:


#Random Forest Clasifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score


# In[888]:


#creatong confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[889]:


#Classification report
print(classification_report(y_test, y_pred))


# In[890]:


### roc_auc_score

FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
### Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[891]:


### ExtraTreesClassifier

clf = ExtraTreesClassifier()
clf.fit(X_train,y_train)       
y_pred=clf.predict(X_test)
et=metrics.accuracy_score(y_test, y_pred)      
et
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score

et=metrics.accuracy_score(y_test, y_pred)
et

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

### classification_report

print(classification_report(y_test, y_pred))

### roc_auc_score

FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - ExtraTreestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[892]:


### Gradient boosting

clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
gb=metrics.accuracy_score(y_test, y_pred)
gb
rfc_cv_score = cross_val_score(clf, X, y, cv=5)
rfc_cv_score

gb=metrics.accuracy_score(y_test, y_pred)
gb

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

### classification_report

print(classification_report(y_test, y_pred))

### roc_auc_score

FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[893]:


### SVC

svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
sv=metrics.accuracy_score(y_test, y_pred)
sv
rfc_cv_score = cross_val_score(svc, X, y, cv=5)
rfc_cv_score

sv=metrics.accuracy_score(y_test, y_pred)
sv

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

### classification_report

print(classification_report(y_test, y_pred))

### roc_auc_score

FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for Gradient BooostingClassifier: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - Gradient BoostingClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[894]:


### Gaussian NB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred=svc.predict(X_test)
nb=metrics.accuracy_score(y_test, y_pred)
nb
rfc_cv_score = cross_val_score(gnb, X, y, cv=5)
rfc_cv_score

nb=metrics.accuracy_score(y_test, y_pred)
nb

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

### classification_report

print(classification_report(y_test, y_pred))

### roc_auc_score

FP, TP, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for RandomForestClassifier: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10,6))
plt.title('Receiver Operating Characteristic - RandomForestClassifier')
plt.plot(FP, TP)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"),
plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize = (10,6))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision-Recall Curve')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

X=et_df
X
y

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=20))
sel.fit(X_train, y_train)

sel.get_support()

features = X_train.columns[sel.get_support()]

X_train.head()

X_train_rfe = sel.transform(X_train)
X_test_rfe = sel.transform(X_test)

def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)

%%time
run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)

%%time
run_randomForest(X_train, X_test, y_train, y_test)

y_train.shape


# In[ ]:


### RandomForestClassifier

r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    l.append(index)
    r.append(key)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)

randf = pd.DataFrame(list(zip(l, f)),columns =['features', 'r_accuracy'])
randf

randf.plot(kind = 'line',
        x = 'features',
        y = 'r_accuracy',
        color = 'green')

### GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)

gb = pd.DataFrame(list(zip(l, f)),columns =['features', 'g_accuracy'])
gb

gb.plot(kind = 'line',
        x = 'features',
        y = 'g_accuracy',
        color = 'blue')

### ExtraTreesClassifier

r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)

ext = pd.DataFrame(list(zip(l, f)),columns =['features', 'e_accuracy'])
ext

ext.plot(kind = 'line',
        x = 'features',
        y = 'e_accuracy',
        color = 'orange')

### SVC

from sklearn.svm import LinearSVC
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    print()
    f.append(acc)

svc = pd.DataFrame(list(zip(l, f)),columns =['features', 'svc_accuracy'])
svc

svc.plot(kind = 'line',
        x = 'features',
        y = 'svc_accuracy',
        color = 'black')

### GaussianNB

from sklearn.naive_bayes import GaussianNB
r=[]
f=[]
k=[]
l=[]
index=0
for key in X_train.columns:
    index=index+1
    r.append(key)
    l.append(index)
    k.append(key)
    X_train1=X_train[r]
    X_test1=X_test[r]
    print('Selected Feature: ', index)
    print('Column Names:',r)
    clf = GaussianNB()
    clf.fit(X_train1, y_train)
    y_pred = clf.predict(X_test1)
    acc=accuracy_score(y_test, y_pred)
    print('Accuracy:',acc)
    f.append(acc)

nb = pd.DataFrame(list(zip(l, f)),columns =['features', 'nb_accuracy'])
nb

nb.plot(kind = 'line',
        x = 'features',
        y = 'nb_accuracy',
        color = 'yellow')

all_alg=randf

all_alg['e_accuracy']=ext['e_accuracy']
all_alg['svc_accuracy']=svc['svc_accuracy']
all_alg['nb_accuracy']=nb['nb_accuracy']
all_alg['g_accuracy']=gb['g_accuracy']

all_alg.plot(kind = 'line',
        x = 'features',
        y = ['nb_accuracy','e_accuracy','svc_accuracy','g_accuracy'])


# In[ ]:


### Generating new features

et_df.columns

et_df["HOMO"] = et_df["B_HOMO-"] + et_df["B_HOMO+"]
et_df["AFFIN"]=et_df["B_e_affin+"] + et_df["A_e_affin+"]
et_df["LUMO"] = et_df["B_LUMO+"] + et_df["B_LUMO-"]
et_df["OS"] = et_df["B_OS"] + et_df["A'_OS"]
et_df["IE"] = et_df["A_IE+"] + et_df["B_IE+"]

et_df.head()


# # IMPLEMENTING PYMATGEN AND ENERGY ABOVE HULL

# In[ ]:


from pymatgen.ext.matproj import MPRester
import pandas as pd
import pymatgen.analysis.phase_diagram as PhaseDiagram
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
API_KEY = "Bfq7dyNTipwl5vS2"
with MPRester(API_KEY) as mpr:
    print(mpr.supported_properties)


# In[895]:


#Thermodynamic stability of dataset
sns.relplot(x="t", y="e_above_hull(meV)", hue="stable", data=df2);

