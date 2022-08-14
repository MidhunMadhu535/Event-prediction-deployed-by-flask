import pandas as pd
import numpy as np
import pickle
df=pd.read_csv("C:\\Users\\user\\Desktop\\Deep Learning\\12 day - Flask\\z vs code\\DeployModel_Flask\\intro\\iris.data")
x=np.array(df.iloc[:,0:4])
y=np.array(df.iloc[:,4:])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.1)

from sklearn.svm import SVC
sv=SVC(kernel='linear').fit(X_train,Y_train)
pickle.dump(sv,open('iri.pkl','wb'))