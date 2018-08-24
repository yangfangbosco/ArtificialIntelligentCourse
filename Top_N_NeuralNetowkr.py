
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import string


dataset = data = pd.read_csv('/Users/yangfang/Downloads/reviews-2.csv',delimiter = '|')


# In[2]:


dataset


# In[3]:


train = dataset[dataset.index % 5 != 0]
test = dataset[dataset.index % 5 == 0]


# In[4]:


train_x = train['text'].tolist()
train_y = train['label'].tolist()
test_x = test['text'].tolist()
test_y = test['label'].tolist()


# In[5]:


train_x


# In[6]:


lemmatizer = nltk.WordNetLemmatizer()


# In[7]:


stopwords = {u'a',
 u'about',
 u'above',
 u'after',
 u'again',
 u'against',
 u'ain',
 u'all',
 u'am',
 u'an',
 u'and',
 u'any',
 u'are',
 u'aren',
 u"aren't",
 u'as',
 u'at',
 u'be',
 u'because',
 u'been',
 u'before',
 u'being',
 u'below',
 u'between',
 u'both',
 u'but',
 u'by',
 u'can',
 u'couldn',
 u"couldn't",
 u'd',
 u'did',
 u'didn',
 u"didn't",
 u'do',
 u'does',
 u'doesn',
 u"doesn't",
 u'doing',
 u'don',
 u"don't",
 u'down',
 u'during',
 u'each',
 u'few',
 u'for',
 u'from',
 u'further',
 u'had',
 u'hadn',
 u"hadn't",
 u'has',
 u'hasn',
 u"hasn't",
 u'have',
 u'haven',
 u"haven't",
 u'having',
 u'he',
 u'her',
 u'here',
 u'hers',
 u'herself',
 u'him',
 u'himself',
 u'his',
 u'how',
 u'i',
 u'if',
 u'in',
 u'into',
 u'is',
 u'isn',
 u"isn't",
 u'it',
 u"it's",
 u'its',
 u'itself',
 u'just',
 u'll',
 u'm',
 u'ma',
 u'me',
 u'mightn',
 u"mightn't",
 u'more',
 u'most',
 u'mustn',
 u"mustn't",
 u'my',
 u'myself',
 u'needn',
 u"needn't",
 u'no',
 u'nor',
 u'not',
 u'now',
 u'o',
 u'of',
 u'off',
 u'on',
 u'once',
 u'only',
 u'or',
 u'other',
 u'our',
 u'ours',
 u'ourselves',
 u'out',
 u'over',
 u'own',
 u're',
 u's',
 u'same',
 u'shan',
 u"shan't",
 u'she',
 u"she's",
 u'should',
 u"should've",
 u'shouldn',
 u"shouldn't",
 u'so',
 u'some',
 u'such',
 u't',
 u'than',
 u'that',
 u"that'll",
 u'the',
 u'their',
 u'theirs',
 u'them',
 u'themselves',
 u'then',
 u'there',
 u'these',
 u'they',
 u'this',
 u'those',
 u'through',
 u'to',
 u'too',
 u'under',
 u'until',
 u'up',
 u've',
 u'very',
 u'was',
 u'wasn',
 u"wasn't",
 u'we',
 u'were',
 u'weren',
 u"weren't",
 u'what',
 u'when',
 u'where',
 u'which',
 u'while',
 u'who',
 u'whom',
 u'why',
 u'will',
 u'with',
 u'won',
 u"won't",
 u'wouldn',
 u"wouldn't",
 u'y',
 u'you',
 u"you'd",
 u"you'll",
 u"you're",
 u"you've",
 u'your',
 u'yours',
 u'yourself',
 u'yourselves'}


# In[8]:


replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
string.punctuation


# In[9]:


def preprocessing(line):
    line = line.replace('<br />', '')   # Remove html tag (<br />)
    line = line.translate(replace_punctuation)     # Remove punctuation
    line = ''.join([i for i in line if not i.isdigit()])
    # Get tokens
    tokens = []
    for t in line.split():
        t = t.lower()
        if t not in stopwords:
            lemma = lemmatizer.lemmatize(t, 'v')
            tokens.append(lemma)
    return ' '.join(tokens)
train_x = [preprocessing(x) for x in train_x]
test_x = [preprocessing(x) for x in test_x]


# In[10]:


train_x


# In[11]:


all_words = []
for line in train_x:
    words = line.split()
    for w in words:
        all_words.append(w)
        
voca = nltk.FreqDist(all_words)


# In[12]:


voca


# In[13]:


from sklearn.neural_network import MLPClassifier


# In[14]:


mnb_model = MLPClassifier()


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


# In[16]:


def train_with_n_topwords(n: int, tfidf=False) -> tuple:
    """
    Train and get the accuracy with different model settings
    Args:
        n: number of features (top frequent words in the vocabulary)
        tfidf: whether do tf-idf re-weighting or not
    Outputs:
        tuple: (accuracy score, classifier, vectorizer)
    """
    topwords = [fpair[0] for fpair in list(voca.most_common(n))]
    
    
    vec = TfidfVectorizer(vocabulary=topwords)
    
    
    # Generate feature vectors
    train_features = vec.fit_transform(train_x)
    test_features  = vec.transform(test_x)
    
    # NB
    mnb_model = MLPClassifier()
    mnb_model.fit(train_features, train_y)
    
    # Test predict
    pred = mnb_model.predict(test_features)
    pred_prob = mnb_model.predict_proba(test_features)
    
    pred_proba = []
    for percent in pred_prob:
        pred_proba.append(percent[1])
    
    
    return [mnb_model, vec, metrics.accuracy_score(test_y, pred), metrics.precision_score(test_y,pred, pos_label='positive'), metrics.recall_score(test_y,pred,pos_label='positive'), metrics.roc_curve(test_y,pred_proba,pos_label='positive')]


# In[17]:




possible_n = [1000 * i for i in range(1, 10)]

model = []

vec = []

accuracies = []
precision = []
recall = []
roc_curve = []
f1 = []

for i, n in enumerate(possible_n):
    things = train_with_n_topwords(n)
    model.append(things[0])
    vec.append(things[1])
    accuracies.append(things[2])
    precision.append(things[3])
    recall.append(things[4])
    roc_curve.append(things[5])
    f1.append(things[3]*things[4]*2/(things[3]+things[4]))
    print("done with i="+ str(i))


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(possible_n, accuracies, label='accuracies')
plt.legend()


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(possible_n, precision, label='precision')
plt.legend()


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(possible_n, recall, label='recall')
plt.legend()


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(possible_n, f1, label='f1')
plt.legend()

