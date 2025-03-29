import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
new_df = pd.read_csv(r"C:\Users\SHUBHAM SHARMA\Desktop\programming\python\AI Course Udemy\NATURAL LANGUAGE PROCESSING\fake news\WELFake_Dataset.csv")

new_df = new_df.fillna(' ')

new_df['content'] = new_df['title']
lemmatizer = WordNetLemmatizer()
def stemming(content):
   stemmed_content = re.sub('[^a-zA-Z]'," ",content)
   stemmed_content = stemmed_content.lower()
   stemmed_content=stemmed_content.split()
   stemmed_content=[lemmatizer.lemmatize(word) for word in stemmed_content if word not in stopwords.words("english")]
   stemmed_content=" ".join(stemmed_content)
   return stemmed_content
new_df['content'] = new_df['content'].apply(stemming)
print(new_df.tail)
new_df['content']
x = new_df['content'].values
y = new_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
train_y_pred = model.predict(x_train)
print("train accuracy:  ",accuracy_score(train_y_pred,y_train))
test_y_pred = model.predict(x_test)
print("train accuracy:  ",accuracy_score(test_y_pred,y_test))
input_data = x_test[1120]
predictor = model.predict(input_data)
if predictor[0]==1:
    print("Fake News")
else:
    print('Real News')
