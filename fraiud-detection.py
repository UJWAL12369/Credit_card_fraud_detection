import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\srish\Downloads\creditcard.csv\creditcard.csv")
print(df)

pd.options.display.max_columns = None

print(df.shape)


print(df.info())


print(df.isna())

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))



df=df.drop(['Time'],axis=1)
print(df)

df=df.drop_duplicates()

print(df.shape)


# not handling imbalanced

# highly imbalnaced


print(df['Class'].value_counts())

print("Hello World")


x=df.drop('Class',axis=1)
print(x)

y=df['Class']


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# what happrns of we do not handle imbalanced data

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()

reg.fit(x_train,y_train)

y_pred1=reg.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))


from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(y_test,y_pred1))

print(recall_score(y_test,y_pred1))

print(f1_score(y_test,y_pred1))


# undesampling

normal=df[df['Class']==0]
fraud=df[df['Class']==1]

print(normal.shape)

print(fraud.shape)


normal_sample=normal.sample(n=473)


newdf=pd.concat([normal_sample,fraud],ignore_index=True)
print(newdf)

X1=newdf.drop('Class',axis=1)
Y2=newdf['Class']

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(X1,Y2,test_size=0.2,random_state=42)

reg1=LogisticRegression()
reg1.fit(xtr,ytr)

ypred2=reg1.predict(xte)

from sklearn.metrics import accuracy_score
print(accuracy_score(yte,ypred2))
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(yte,ypred2))

print(recall_score(yte,ypred2))

print(f1_score(yte,ypred2))


# dec tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(xtr,ytr)

y_pred2=dt.predict(xte)

print(accuracy_score(yte,y_pred2))
print(precision_score(yte,y_pred2))
print(recall_score(yte,y_pred2))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(xtr,ytr)
ypred3=rf.predict(xte)
print(accuracy_score(yte,ypred3))
print(precision_score(yte,ypred3))
print(recall_score(yte,ypred3))


final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(yte,ypred2)*100,
                     accuracy_score(yte,y_pred2)*100,
                     accuracy_score(yte,ypred3)*100
                    ]})
print(final_data)


# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,5))

# sns.barplot(x='Models', y='ACC', data=final_data)

# plt.xticks(rotation=45)
# plt.title("Model Accuracy Comparison")
# plt.xlabel("Models")
# plt.ylabel("Accuracy")

# plt.show()


# oversampling balance an imbalanced dataset by increasing the number of samples in the minority class.

x=df.drop('Class',axis=1)
print(x)

y=df['Class']

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(x,y)
print(y_resampled.value_counts())

xtr1,xte1,ytr1,yte1=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42)

reg1.fit(xtr1,ytr1)
yu1=reg1.predict(xte1)

print(accuracy_score(yte1,yu1))
print(precision_score(yte1,yu1))

# decsion tree

dt.fit(xtr1,ytr1)
yu2=dt.predict(xte1)
print(accuracy_score(yte1,yu2))
print(precision_score(yte1,yu2))
      
# ramdom forest c;assifier
rf.fit(xtr1,ytr1)
yu3=rf.predict(xte1)
print(accuracy_score(yte1,yu3))
print(precision_score(yte1,yu3))


rf1=RandomForestClassifier()
rf1.fit(X_resampled,y_resampled)




