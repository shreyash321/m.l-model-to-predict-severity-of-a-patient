import pandas as ss 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
d=ss.read_csv('https://www.stats.govt.nz/assets/Uploads/Serious-injury-outcome-indicators/Serious-injury-outcome-indicators-2000-20/Download-data/serious-injury-outcome-indicators-2000-2020-CSV.csv')
print(d)
d['Moving average']=d['Type']=='Moving average'
d['Assualt']=d['Cause']=='Assault'
d['Children']=d['Population']=='Children'
p=d[['Data_value','Lower_CI','Upper_CI']].values
k=d[['Data_value','Lower_CI','Upper_CI','Children','Assualt','Moving average']].values
l=d['Severity'].values
x_train,x_test,y_train,y_test = train_test_split(k,l)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
#print(model.predict([[41,28.45,53.54,True,False,False]]))
i=int(input())
x=x_test[i]
print('Results of Logistic Regression :--')
print('actual output--',y_test[i])
print('prediction',model.predict([x]))
print(model.score(x_test,y_test))
rf=RandomForestClassifier(n_estimators=20,random_state=111)
rf.fit(x_train,y_train)
first_row=x_test[i]
print('Results of random forest :--')
print('actual output--',y_test[i])
print('prediction',rf.predict([first_row]))
print(rf.score(x_test,y_test))
#pf = ss.DataFrame(np.random.randn(1000, 5), columns=list("ABCDE"))
pf = ss.DataFrame(p,columns=list("ABC"))
pf = pf.cumsum()
plt.figure()
pf.plot()
