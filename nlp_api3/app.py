

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pickle
from function import clean
import csv
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import os
import csv
# load the model from disk
filename = 'model.pkl'
#clean=pickle.load(open('data_clean.pkl', 'rb'))
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tv_tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	return render_template('index.html')


@app.route('/data',methods=['GET','POST'])
def data():
    if request.method == 'POST':
        f = request.form["textfile"]
        data=[]
        with open(f,'r') as file:
            
            q1=file.read()
            
            data.append(q1)
            data=pd.DataFrame(data,columns=['text'])
            data= data.apply(clean)
            vect = cv.transform(data).toarray()
            my_prediction = clf.predict(vect)
            
                
            
        return render_template('data.html',data=my_prediction)

        



if __name__ == '__main__':
	app.run(debug=True)