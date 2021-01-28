import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

emails =pd.read_csv("emails.csv")
emails.head()
emails.shape
emails.dtypes
emails.info()
emails.columns
emails['Class'].value_counts()

email_data = emails[['content', 'Class']]
email_data.head()
print(email_data["Class"].value_counts())
sns.countplot(email_data["Class"])
email_data.groupby('Class').describe()
email_data['content_count']=email_data['content'].apply(lambda x: len(str(x)))
email_data.head()
email_data['content_count'].describe()
email_data[email_data['content_count']==272036]['content'].iloc[0]
email_data[email_data['content_count']==1]['content'].iloc[0]

import re

# Removal of "n\" characters
email_data["content_w_space"]=email_data["content"].replace('\n'," ",regex=True)
email_data["content_w_space"].head(10)
def to_lower(text):
    result = str(text).lower()
    return result
email_data["content_low"]=email_data["content_w_space"].apply(lambda x: to_lower(x))
email_data["content_low"].head()
def remove_special_characters(text):
    #result = re.sub("[^A-Za-z0-9]+"," ", text)
    result =  re.sub(r'[^a-zA-Z]', ' ', text)
    return result
email_data["content_wsch"]=email_data["content_low"].apply(lambda x: remove_special_characters(x))
email_data["content_wsch"].head()

def removal_hyperlinks(text):
    result =  re.sub(r"http\\S+", " ", str(text))
    return result

email_data["content_whl"]=email_data["content_wsch"].apply(lambda x: removal_hyperlinks(x))
email_data["content_whl"].head()

#  Removal of whitespaces:
def removal_whitespaces(text):
    result =  re.sub(' +', ' ', text)
    return result
email_data["content_wws"]=email_data["content_whl"].apply(lambda x: removal_whitespaces(x))
email_data["content_wws"].head()

# 7. Removal of stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
#print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

def removal_stopwords(text):
    word_tokens = word_tokenize(text)  
    filtered_sentence = []
    a_row=""
    for a_word in word_tokens:
        if a_word not in stop_words:
            filtered_sentence.append(a_word)
            a_row = " ".join(filtered_sentence)
    return a_row

email_data["content_w_sw"]=email_data.content_wws.apply(lambda x: removal_stopwords(x))
email_data["content_w_sw"].head()

# Text normalization, lemmatization

from nltk import WordNetLemmatizer

lemma = WordNetLemmatizer()
def lemmatization(text):
    word_tokens = word_tokenize(text) 
    a_array=[]
    a_string = ""
    for a_word in word_tokens:
               
        a_lemma = lemma.lemmatize(a_word,pos = "n")
        a_lemma1 = lemma.lemmatize(a_lemma, pos="v")
        a_lemma2 = lemma.lemmatize(a_lemma1, pos="a")
   
        a_array.append(a_lemma2)
        
        a_string = " ".join(a_array)
    return a_string



email_data["content_lemma"]=email_data.content_w_sw.apply(lambda x: lemmatization(x))
email_data["content_lemma"].head()

email_data["content_cleaned"]=email_data["content_lemma"]
email_data["length"] = email_data["content_cleaned"].apply(lambda x: len(x))
email_data["length"].describe()

data = email_data[["content_cleaned","Class","length"]]
data.head()
len(data)
# Checking for duplicates

duplicate_records = data[data.duplicated()] 
duplicate_records.head(5)
len(duplicate_records)

data =  data.drop_duplicates() # keeping the first value
data.head()
data.shape

#data.to_csv("email_data_final.csv")

# Visualization and Word Cloud

# Target Variable Class
    
print(data["Class"].value_counts())
sns.countplot("Class", data = data)
data['length'].plot(bins=50,kind='hist')
data['length'].describe()
email_abusive= data[(data["Class"]=="Abusive")]
email_abusive.shape
email_abusive.content_cleaned[0:5]
email_non_abusive= data[(data["Class"]=="Non Abusive")]
email_non_abusive.shape

final_email_abusive=""
abusive_email =[]
for text in email_abusive["content_cleaned"]:
    abusive_email.append(text)
    final_email_abusive =  "".join(abusive_email)
final_email_abusive

from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS) 

wordcloud_abusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_abusive)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_abusive_words,interpolation="bilinear")

final_email_nonabusive=""
nonabusive_email =[]
for text in email_non_abusive["content_cleaned"]:
    nonabusive_email.append(text)
    final_email_nonabusive =  "".join(nonabusive_email)
final_email_nonabusive

wordcloud_nonabusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_nonabusive)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_nonabusive_words,interpolation="bilinear")


data.columns

from nltk.tokenize import word_tokenize
data["content_tokenized"]= data["content_cleaned"].apply(lambda x: word_tokenize(x) )

data["content_tokenized"].head()
data.columns


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X1=cv.fit(data["content_cleaned"])
X=cv.fit_transform(data["content_cleaned"])

print(len(X1.vocabulary_))

a_email=data['content_cleaned'][4]
a_email

a_email_vector=X1.transform([a_email])
print(a_email_vector)
print(a_email_vector.shape)

print(X1.get_feature_names()[24004])
print(X1.get_feature_names()[43987])

mails= X1.transform(data['content_cleaned'])
mails.shape

print('Shape of Sparse Matrix: ',mails.shape)
print('Amount of non-zero occurences:',mails.nnz)


sparsity =(100.0 * mails.nnz/(mails.shape[0]*mails.shape[1]))
print('sparsity:{}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(mails)
tfidf_a_email = tfidf_transformer.transform(a_email_vector)
print(tfidf_a_email.shape)

print(tfidf_transformer.idf_[X1.vocabulary_['gamble']])
print(tfidf_transformer.idf_[X1.vocabulary_['asshole']])
print(tfidf_transformer.idf_[X1.vocabulary_['excelr']])
print(tfidf_transformer.idf_[X1.vocabulary_['ect']])
print(tfidf_transformer.idf_[X1.vocabulary_['make']])
print(tfidf_transformer.idf_[X1.vocabulary_['lavorato']])
print(tfidf_transformer.idf_[X1.vocabulary_['problem']])
print(tfidf_transformer.idf_[X1.vocabulary_['go']])


emails_tfidf = tfidf_transformer.transform(mails)
print(emails_tfidf.shape)
emails_tfidf

# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
data['Class']= label_encoder.fit_transform(data['Class']) 
  
data['Class'].unique()

data["Class"].value_counts()

X_data = emails_tfidf
#X_data = X_feature
y =data["Class"]
X_data.shape

y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=0)

X_train.shape, y_train.shape, X_test.shape,  y_test.shape

print(y_train.value_counts())
sns.countplot(y_train)

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=0)

X_train_bal, y_train_bal = sm.fit_resample(X_train,y_train)

X_train_bal.shape, y_train_bal.shape

print(y_train_bal.value_counts())
sns.countplot(y_train_bal)


y_train_bal

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train_bal,y_train_bal)

# Train Accuracy
y_train_pred = xgb_model.predict(X_train_bal)
train_accur_xgb =accuracy_score(y_train_bal,y_train_pred)

# Test accuracy
y_test_pred = xgb_model.predict(X_test)
test_accu_xgb =accuracy_score (y_test,y_test_pred)

train_accur_xgb, test_accu_xgb

recall_score_xgb= recall_score(y_test,y_test_pred)
recall_score_xgb

precision_score_xgb = precision_score(y_test,y_test_pred)

precision_score_xgb

f1_score_xgb = f1_score(y_test,y_test_pred)
f1_score_xgb

print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))
pd.crosstab(y_test.values.flatten(),y_test_pred)



example1 = {"You are a monkey shit"}
df=pd.DataFrame(example1)
result1= xgb_model.predict(example1)
print(result1)

example2=["i am very happy i want to travel"]
result2 = xgb_model.predict(example2)
print(result2)

ex3=["The violence of the storm caused great fear."]
result3= xgb_model.predict(ex3)
print(result3)










