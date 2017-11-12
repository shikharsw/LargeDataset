import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer


stopw = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 
         'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 
         'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am',
         'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 
         'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself',
         'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 
         'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 
         'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 
         'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 
         'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 
         'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}



def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopw]
    return " ".join(filtered_words)

words = dict()
labels = dict()
x = []
y = []
#with open('./full/full_train.txt') as file:
with open('./verysmall/verysmall_train.txt') as file:
    i=1
    j=0
    k=0
    for line in file:
        if(i<4):
            i=i+1
            continue
        a = line.split('\t')
        #print(line)
        x.append(preprocess(a[1]))
        y.append(a[0].strip())
        for l in a[0].strip().split(','):
            if(not(l in labels.keys()) ):
                labels[l] = k;
                k=k+1
        for w in preprocess(a[1]).split(' '):
            if(not(w in words.keys()) ):
                words[w] = j
                j=j+1

xtest = []
ytest = []
#with open('./full/full_test.txt') as file:
with open('./verysmall/verysmall_test.txt') as file:
    i=1
    for line in file:
        if(i<4):
            i=i+1
            continue
        a = line.split('\t')
        #print(line)
        xtest.append(preprocess(a[1]))
        ytest.append(a[0].strip())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# gives one hot vector matrix
def pred(x,w):
    wx = np.dot(x,w)
    pred = sigmoid(wx)
    arg = pred.argmax(axis=1)
    predval = np.eye(50)[arg]
    return predval
    
def updatew(w,x,y,rate,lamda):
    w = w + rate*( np.matmul(x.T,(y-pred(x,w))) + lamda*w)
    return w
    
def acc(pred,true):
    c = np.sum(pred.argmax(axis=1)==true.argmax(axis=1))
    return c/np.shape(true)[0]

# training
cur = 0
batchsize = 256
wt = np.random.rand(len(words),50)
print("Training started")
for epoch in range(2):
    count = 0
    tacc = 0
    for j in range(0,int(len(x)/10)):
        xbatch = np.zeros((batchsize,len(words)))
        ybatch = np.zeros((batchsize,50))
        for i in range(batchsize):
            for w in x[i].split():
                xbatch[i][words[w]] = 1
            for l in y[i].split(','):
                ybatch[i][labels[l]] = 1

        wt = updatew(wt,xbatch,ybatch,0.1,0.4)
        ypred = pred(xbatch,wt)
        batchacc = acc(ypred,ybatch)
        tacc = tacc + batchacc
        count = count +1
        cur = cur + batchsize
    trainacc = tacc/count
    print("epoch",epoch,"acc",trainacc)
print("Training end")

# testing/prediction
cur = 0
batchsize = 10
avgacc = 0
count = 0
for j in range(0,int(len(xtest)/10)):
#for j in range(0,10):
    xtestbatch = np.zeros((batchsize,len(words)))
    ytestbatch = np.zeros((batchsize,50))
    for i in range(batchsize):
        for w in xtest[i].split():
            xtestbatch[i][words[w]] = 1
        for l in ytest[i].split(','):
            ytestbatch[i][labels[l]] = 1

    ypred = pred(xtestbatch,wt)
    batchacc = acc(ypred,ytestbatch)
    print(batchacc)
    cur = cur + batchsize
    count = count + 1
    avgacc = avgacc + batchacc

avgacc = avgacc/count
print("test acc",avgacc)

