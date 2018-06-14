
# coding: utf-8

# In[1]:


# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import csv


# In[2]:


training_data = []
with open('../data/total.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        training_data.append({"hscode" : row[0], "item" : row[1] })

print ("%s sentences in training data" % len(training_data))


# In[3]:


words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['item'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['hscode']))
    # add to our classes list
    if pattern['hscode'] not in classes:
        classes.append(pattern['hscode'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "hscodes", classes)
print (len(words), "unique stemmed words", words)


# In[4]:


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# In[5]:



# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

with tf.device('/cpu:0'):
	# Define model and setup tensorboard
	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
	# Start training (apply gradient descent algorithm)
	model.fit(train_x, train_y, n_epoch=50, batch_size=8, show_metric=True)
	model.save('model.tflearn')


# In[ ]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[ ]:


print(words)
p = bow("corn meal", words)
print (p)
print (classes)


# In[ ]:


#print(model.predict([p]), classes)
a =np.squeeze(np.asarray(model.predict([p]))) 
print(a)
# for i, v in zip(a, classes):
# print(i, v)
from itertools import zip_longest
import operator
import collections
mydict = dict(zip_longest(a, classes))

sorted_mydict = collections.OrderedDict(sorted(mydict.items(), reverse=True)[:10])
print(sorted_mydict)


# In[ ]:


# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

