# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:03:47 2021

@author: uddish lalaji
"""

import pandas as pd
import numpy as np
import os
import glob,re,string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
stopwords=stopwords.words('english')



def clean(text):
        text=" ".join(str(text).split('\n'))
        text=re.sub(r'Agent|Customer','',str(text),flags=re.I|re.M)
        text=re.sub(r'noise|silence|uh-huh','',str(text),flags=re.I|re.M)
        text=re.sub(r'[0-9]+','',str(text))
        text=text.lower()
        text=re.sub(r'\.|\[|\]|okay|um|yes|uh|all|right|hello|_|-\|\'','',str(text),flags=re.I|re.M)
        text=re.sub(r'\'ve',' have',str(text))
        text=re.sub(r"let's"," let us",str(text))
        text=re.sub(r'\'m', ' am',str(text))
        text=re.sub(r'\'ll',' will',str(text))
        text=re.sub(r'\'s',' is',str(text))
        text=re.sub(r'n\'t',' not',str(text))
        text=re.sub(r'we|hey|how|you|doing|what|is|are|so|well','',str(text),flags=re.I|re.M)
        text=re.sub("\\bi\\b|\\ba\\b|\\bll\\b|\\bthe\\b|\\bhi\\b|\\bam\\b|\\bfine\\b|\\bhave\\b|\\br\\b|\\boh\\b\
        |\\bof\\b|\\bthem\\b|\\babout\\b|\\bto?o\\b|\\bs\\b|\\bminute\\b|\\bin\\b|\\bout\\b|\\boh?h\\b",' ',str(text),flags=re.I|re.M)
        text=re.sub(r'\'|:|\d','',str(text))
        text=re.sub(r' +',' ',str(text))
            
           
        nopunct=re.sub('['+"\\".join(string.punctuation)+']','',text)
        nopunct="".join(nopunct)
        nostp=[word for word in nopunct.split() if word not in stopwords]
        lemma=" ".join([lemmatizer.lemmatize(w) for  w in nostp])
            
        
        return lemma
         
