# Libraries ..
import tweepy as tw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
import string
import re
from textblob import TextBlob, Word
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer,TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
# Twitter Api
# define keys
consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""
#Twitter connect code
baglanti = tw.OAuthHandler(consumer_key, consumer_secret)
baglanti.set_access_token(access_token, access_token_secret)
api=tw.API(baglanti,wait_on_rate_limit=True)
#%%
all_tweets = []

#choose = input("choose:")
search_words = input("Enter search word:")
number_of_search_words = int(input("enter number of search word:"))
tweets = tw.Cursor(api.search, q=search_words, lang="en" ).items(number_of_search_words)
    
all_tweets = [tweet.text for tweet in tweets]       
Out = {'Search_words': search_words ,'Tweets': all_tweets  }

print(search_words)  
for tweet in tweets:
    print(tweet.text) 
    print ("**")
df = pd.DataFrame(Out, columns=['Search_words','Tweets'])

df.to_csv("out.csv", encoding='utf-8', index=False)
#%%
data = pd.read_csv("out.csv")
print('Count total number of Tweets')
print('*****')
print(data.head(5))
data.tail(5)
#%%
#  remove punctuations
def remove_punctuations(text):
    text = ' '.join([i for i in text if i not in frozenset(string.punctuation)])
    return text

stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('oh')
print(stop_words)

# remove stopword
def remove_stopword(text):
    words = [w for w in text if w not in stop_words]
    return words

# remove url and user
def remove_url_and_user(text):
    text = re.sub('@[A-Za-z0-9]+','',text)
    text = re.sub('https?://[A-Za-z0-9./]+','', text)
    return text

# remove digits
def remove_digits(text):
    text = re.sub('[0-9]+','', text)
    text = re.sub('_[A-Za-z0-9./]+','', text)
    return text
#%%
data['Tweets_2'] = data['Tweets'].apply(remove_url_and_user)
data['Tweets_3'] = data['Tweets_2'].str.replace('RT','') # RT ifadesi silindi
data['Tweets_3'] = data['Tweets_3'].apply(lambda x: x.lower()) # bütün harfler küçük

tokenizer = RegexpTokenizer(r'\w+')
#datafr['Tweets'] = datafr['Tweets'].apply(lambda x: word_tokenize(x))
data['Tweets_4'] = data['Tweets_3'].apply(lambda x: tokenizer.tokenize(x))

data['cleaned'] = data['Tweets_4'].apply(remove_stopword) # çok kullanılan kelimeler silindi.
data['cleaned'] = data['cleaned'].apply(remove_punctuations) # noktalama işaretleri silindi.
data['cleaned'] = data['cleaned'].apply(remove_digits) # sayısal değerleri silindi.

data.head(20)
#%%
print(data['cleaned'].loc[1:20])
#%%
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data['temiz'] = data['cleaned'].apply(lemmatize_text)
data['temiz'].loc[1:10]
#%%
# TF_IDF (Term Frequency — Inverse Document Frequency) 
vectorizer = TfidfVectorizer()
#corpus = list(seri)
sentence_list = []
for i in data['cleaned']:
    sentence_list.append(i)
corpus = list(sentence_list)
x = vectorizer.fit(corpus)
print(vectorizer.get_feature_names())
print(x.vocabulary_)
#%%
X = vectorizer.transform(corpus)
print(X.shape)
print(X)
print(X.toarray())
#%%
df = pd.DataFrame(X.toarray(), columns= vectorizer.get_feature_names())
df.T
#%%
# Number of all words  in tweets
word_list = []

for tweet in data['cleaned']:
    for part in tweet.split(' '):
        word_list.append(part)

seri = pd.Series(data=word_list)
print("Count total number of words in text: ", seri.count())
seri.value_counts()
#%%
# WordCloud
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                        background_color='#f2f2f2',
                        width=550,
                        height=374
                     ).generate(" ".join(seri))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#%%
# Sentiment Analysis with Bar Plot
dataframe = pd.read_csv("out.csv")

def percantage(part, whole):
    return 100 * float(part)/float(whole)

positive = 0
negative = 0
neutral = 0
polarity = 0
search_term = dataframe['Search_words'][1]
count = dataframe['Tweets'].count()    
for tweet in dataframe['Tweets']:
    
    analysis = TextBlob(tweet)
    polarity += analysis.sentiment.polarity
    
    if analysis.sentiment.polarity == 0:
        neutral += 1
    elif analysis.sentiment.polarity >0.00:
        positive += 1
    elif analysis.sentiment.polarity< 0.00:
        negative += 1

positive = percantage(positive, count)
neutral = percantage(neutral, count)
negative = percantage(negative, count)
polarity = percantage(polarity, count)

positive = format(positive, '.2f')
neutral = format(neutral, '.2f')
negative = format(negative, '.2f')

print('Neutral:'+ neutral + ' Positive:' + positive + ' Negative:'+negative )

sizes = ["positive", "neutral", "negative"]
values = [positive, neutral, negative]

dictionary = {"Sizes": sizes, "Values": values}
df = pd.DataFrame(dictionary, columns=['Sizes','Values'])
df['Values'] = df.Values.astype(float)

f,ax = plt.subplots(figsize=(10,5))
sns.set(style="darkgrid", palette="Set1", font_scale=2)
sns.barplot(df["Sizes"],df["Values"],data=df, hue="Values")
plt.show()
print('The number of positive/negative/neutral words in text')

#%%   Sentiment Analysis with Pie Chart
dataframe = pd.read_csv("out.csv") 
def percantage(part, whole):
    return 100 * float(part)/float(whole)

positive = 0
negative = 0
neutral = 0
polarity = 0
search_term = dataframe['Search_words'][1]
count = dataframe['Tweets'].count()    
for tweet in dataframe['Tweets']:
    #print(tweet)   
    analysis = TextBlob(tweet)
    polarity += analysis.sentiment.polarity
    
    if analysis.sentiment.polarity == 0:
        neutral += 1
    elif analysis.sentiment.polarity >0.00:
        positive += 1
    elif analysis.sentiment.polarity< 0.00:
        negative += 1

positive = percantage(positive, count)
neutral = percantage(neutral, count)
negative = percantage(negative, count)
polarity = percantage(polarity, count)

positive = format(positive, '.2f')
neutral = format(neutral, '.2f')
negative = format(negative, '.2f')

print('Kaç kişi ' + search_term + ',' + str(count) + 'Tweets.')

if polarity == 0:
    print('Notr')
elif polarity > 0.00:
    print('Positif')
elif polarity < 0.00:
    print('Negatif')

    
fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect="equal"))
explode = (0.1, 0, 0)

labels = ['Positive ['+str(42.40)+'%]', 'Negative ['+str(14.32)+'%]', 'Notr ['+str(43.28)+'%]']
sizes = [42.40, 14.32, 43.28]

patches , texts , a=ax.pie(sizes,colors=sns.color_palette('tab10'), explode=explode, autopct='%1.2f%%',
        shadow=True, startangle=90) 
plt.title(search_term + ', ' + '10000' + ' Tweets.', fontsize=17)

ax.legend(patches, labels,
          loc="center left",
          fontsize = 16,
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(a, size=14, weight="bold")
#plt.axis("equal")
plt.tight_layout()
