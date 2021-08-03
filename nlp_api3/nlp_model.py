# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pathlib import Path
import os,glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
#from sklearn.pipeline import Pipeline
from function import clean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import pickle
#import nltk
#nltk.download('wordnet')
#stopwords=stopwords.words('english')

df1 = pd.read_csv('D:/tagging_test/tagging_test/metadata/mapping_conv_topic.train.txt', sep=" ", header=None, 
                 names=["row_number", "lable"])
df1 = df1.sort_values(by=['row_number'], ascending=True).reset_index(drop=True)
#os.chdir(r"D:/tagging_test/tagging_test")
#filenames = [i for i in glob.glob("*.txt")]


files = Path('D:/tagging_test/tagging_test').glob('*.txt')

text = list()

for file in files:
    text.append(file.read_text())
    
df2=pd.DataFrame(text,columns =['text'])

df=pd.concat([df1,df2], axis=1, ignore_index=True)
df.columns =['row_number', 'label', 'text']


df=df.drop(['row_number'], axis = 1)

#df['label']=df['label'].map({'Family Finance': 0, 'Job Benefits': 1,'Taxes':2,'Credit Card':3,'Budget':4,'Bank Bailout':5})
cv=TfidfVectorizer()
z=clean
X=df['text'].apply(clean)
x=cv.fit_transform(X)
y=df['label']
pickle.dump(cv, open('tv_tranform.pkl', 'wb'))
pickle.dump(z,open('data_clean.pkl', 'wb'))
#pickle.dumps(z)
 

import random 
random.seed(40)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

classifier=RandomForestClassifier(n_estimators=500,random_state=111)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
pickle.dump(classifier, open('model.pkl', 'wb'))

    
    
#print(accuracy_score(y_test,y_pred))
#print(classification_report(y_test,y_pred))
