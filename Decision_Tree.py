
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


topwords = [fpair[0] for fpair in list(voca.most_common(3000))]


# In[14]:


topwords


# In[15]:



from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


tf_vec = TfidfVectorizer()


# In[17]:


tf_vec.fit(topwords)


# In[18]:


tf_vec.get_feature_names()


# In[19]:


train_features = tf_vec.transform(train_x)


# In[20]:


train_features.shape


# In[21]:


test_features = tf_vec.transform(test_x)


# In[22]:


test_features.shape


# In[23]:


from sklearn import tree


# In[24]:


clf = tree.DecisionTreeClassifier()


# In[25]:


import time

start = time.time()
clf = clf.fit(train_features, train_y)
end = time.time()

print("model trained in %f seconds" % (end-start))


# In[26]:


# Predict test
pred = clf.predict(test_features)
print(pred)


# In[27]:


# Metrics
# metrics.accuracy_score(y_true, y_pred)
from sklearn import metrics
accuracy_test = metrics.accuracy_score(pred,test_y)
print(accuracy_test)


# In[28]:


# Predict train
pred_train = clf.predict(train_features)
print(pred)


# In[29]:


# Metrics
# metrics.accuracy_score(y_true, y_pred)
accuracy_train = metrics.accuracy_score(pred_train,train_y)
print(accuracy_train)


# In[30]:


# Use keyword arguments to set arguments explicitly
print(metrics.classification_report(y_true=test_y, y_pred=pred))


# In[31]:


# Predict a new sentence
# vectorizer needs to be pre-fitted
# At the end of the project, the function signature should be something like:
# predict_new(sentent: str, vec, model) -> str

def predict_new(sentence: str):
    sentence = preprocessing(sentence)
    features = tf_vec.transform([sentence])
    pred = clf.predict(features)
    return pred[0]


# In[32]:



predict_new("There worst yet: i'm a pretty big fan of limp bizkit been listening to them for a few years was very disapointed with this cd. wouldnt reccomened it. try on of there other cds instead")
#neg


# In[33]:


predict_new("Good Instructional Video: This is actually a really good DVD. It teaches you how to do the moves in a fun way. Highly recommende")
#pos


# In[34]:


predict_new("A waste of money: I had to listen to this by myself. I wouldn't want to subject this 'music' to others. It's not that I'm new to 'noise/fuzz' pop. I own and enjoy many CDs by Sonic Youth My Bloody Valentine Dinosaur Jr. but this band just lacks the ability to put together listenable tunes. I had to sell this sorry collection of 'tunes' after 3 or 4 listens. I gave it a chance but each time I listened to it I lost even more interest.")

#neg


# In[35]:


predict_new("Probably Lentz's most engaging piece: Missa Umbrarum the title piece uses an interesting process: A number of 30-second electronic delays are used to slowly build up to the final 'destination' for each phrase of the Mass. The nonvocal music comes entirely from crystal glasses (as per a glass harmonica) played by the vocalists. On the first repetition of a phrase the lowest notes of the section are supplied and then the singers drink from the glasses before adding the next layer. In the end each phrase is built from half-a-dozen or more layers turning the handful of vocalists (and glasses) into a full choir. Despite the technique the final result does not sound dense and muddy; an ethereal nature is retained throughout.")

#pos


# In[36]:


predict_new("Easy on your skin and removes make up easily.: I first found these at Trader Joe's and picked them up on a whim. I was pleasantly surprised at how well they removed my makeup and left my skin feeling clean and cared for and have often followed with moisturizer after a long day at work so I could at least get my makeup off before I went to bed. Of course for days that I have more time and energy I can follow with my normal cleansing routine.I've tried a few other brands but have always returned to Comodyne's as the other brands would start to irritate my skin even in a weeks time and I've never experienced any irritation with Comodyne's.I ordered the 3-pack this time but will order the 6-pack next time for the value and would definitely buy it through the Subscribe & Save program if it were offered.")
#pos


# In[40]:


def draw_roc(clf):
    probs = clf.predict_proba(test_features)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(test_y, preds, pos_label='positive')
    roc_auc = metrics.auc(fpr, tpr)
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt


# In[41]:


plt = draw_roc(clf)


# In[39]:


plt.show()

