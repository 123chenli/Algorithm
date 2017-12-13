from numpy import *

'''
1、函数loadDataSet()
2、创建数据集，这里的数据集是已经拆分好的单词组成的句子，表示的是某论坛的用户评论，标签1表示这个是骂人的
  createVocabList(dataSet)
3、找出这些句子中总共有多少单词，以确定我们词向量的大小
  setOfWords2Vec(vocabList, inputSet)
4、将句子根据其中的单词转成向量，这里用的是伯努利模型，即只考虑这个单词死否存在
  bagOfWords2VecMN(vocabList, inputSet)
5、这个是句子转成向量的另一种模型，多项式模型，考虑某个词的出现次数
  trainNB0(trainMatrix, trainCatergory)
6、计算P(i)和P(w[i]C[1]和P(w[i]|C[0]),这里有两个技巧，一个是开始的分子分母没有全部初始化为0是为了防止
  其中一个概率为0导致整体为0，另一个是后面乘用对数防止因为精度问题结果为0
  classifyNB(vec2Classify, p0Vec, p1Vec, pClass1)
7、根据贝叶斯公式p(ci|w)=p(w|ci)p(ci)/p(w)计算这个向量属于两个集合中哪个的概率高
'''


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建一个带有所有单词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    retVocabList = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVocabList[vocabList.index(word)] = 1
        else:
            print('word', word, 'not in dict')
    return retVocabList


# 另一种模型
def bagOfWords2VecMN(vocabList, inputSet):
    returenVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returenVec[vocabList.index(word)] += 1
    return returenVec


def trainNB0(trainMatrix, trainCatergory):
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCatergory) / float(numTrainDoc)
    # 防止多个概率的成绩当中的一个为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDoc):
        if trainCatergory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix)
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


def main():
    testingNB()


if __name__ == '__main__':
    main()



