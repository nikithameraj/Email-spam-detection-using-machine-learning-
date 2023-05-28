import pickle
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# removing the html tags
def __clean_html(text):
    clean=re.compile('<.*?>')
    cleantext=re.sub(clean,'',text)
    return cleantext
    
# first round of cleaning
def __clean_text(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    
    return text

def predict(message):
    
    model_ = pickle.load(open("model.pkl", 'rb'))
    vectorizer_ = pickle.load(open("cv.pkl", 'rb'))
    
    test=__clean_html(message)
    test=__clean_text(test)
    
    test = vectorizer_.transform([test])

    # column_names = ['EmailText']
    # df = pd.DataFrame(test, columns=column_names)

    output = model_.predict(test)[0]
    
    if output == "ham":
        output = 0
    else:
        output = 1

    return output
