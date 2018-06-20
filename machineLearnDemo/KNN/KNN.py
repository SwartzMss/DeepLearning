#-*- coding:utf-8 -*-
from numpy import  *
import  operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#创建一个最基本的训练数据
def CreateDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return  group,labels

#inX ->检测数据 dataSet->训练集 labels->训练集里面的结果  k->代表最近的个数
def Classify0(inX,dataSet,labels,K):
    # shape(0) 代表的是行数 shape[1]代表的是列数
    dataSetSize = dataSet.shape[0]
    #tile就是利用inX作为元素 生成一个dataSetSize行1列的矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # ** 代表的是乘方 例子3**2 =9
    sqDiffMat = diffMat**2
    #sum(axis=0)列和 ，sum(axis=1)行和
    sqDistances = sqDiffMat.sum(axis = 1)
    #distances 得到的就是距离
    distances = sqDistances**0.5
    #进行排序 ，返回的sortedDistIndicies 是一个索引的数组
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(K):
        #找到前K个距离较近的结果值
        Votelabel = labels[sortedDistIndicies[i]]
        # 获取字典里面的结果的个数 找不到就默认为0
        classCount[Votelabel] = classCount.get(Votelabel,0) + 1
        #这里的话 用classCount的第一个元素进行排序 ，并且不翻转
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fd = open(filename)
    arrayLines = fd.readlines()
    #获取文件里面的行数
    numOfLines = len(arrayLines)
    #创建对应的空数组
    returnMat = zeros((numOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        #去掉一行数据里面的空
        line = line.strip()
        #以制表符作为分隔符获取一个list
        listFromLine = line.split('\t')
        #将前面三个特征值存放到对于的数组里面
        returnMat[index,:] = listFromLine[0:3]
        #将最后的分类结果放到另外的数组里面
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return  returnMat,classLabelVector

#归一化操作
def autoNorm(dataSet):
    #列里面的最小值 和最大值 同样的min(1)就是行里面的最小值
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    #取得最大与做小的差值
    ranges = maxvals - minvals
    #构建一个空的数组
    normDataSet = zeros(shape(dataSet))
    #行数
    m = dataSet.shape[0]
    #减去最小值
    normDataSet = dataSet - tile(minvals,(m,1))
    #上面的结果除以最大和最小的差 newvalue = （oldvalue - min）/（max -min）
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet,ranges,minvals

#测试程序  hoRatio 表示拿来测试数据的百分比 0.5 跟0.1 比较的话可以明显看出最后的结果准确性的差异
def datingClassTest():
    hoRatio = 0.50
    #从文件里面读取训练集 前面的是特征 后面的分类结果
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    #将前面的结果归一化处理
    normMat, ranges, minVals=autoNorm(datingDataMat)
    # 行数
    m=normMat.shape[0]
    #用于测试的数据 行数*0.5 就是拿出一半数据来测试算法的准确性
    numTestVecs=int(m * hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=Classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount+=1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

#将图像信息转换为向量数据 32*32 = 1*1024
def img2Vector(filename):
    #初始化一个 1行1024列的数组
    returnVec = zeros((1,1024))
    fd = open(filename)
    #一共32行 每行读取一次
    for i in range(32):
        lineStr = fd.readline()
        # 每行的话 也有32个数据
        for j in range(32):
            returnVec[0,32*i +j] = int(lineStr[j])
    return  returnVec

def handwritingClassTest():
    hwLabels = []
    #列出训练数据集里面的全部txt
    trainingFileList = listdir('trainingDigits')
    #得到训练集的个数
    m = len(trainingFileList)
    #为每个训练标本都建立一个1024列的行
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        #这里拿到了真正的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :]=img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        #拿到测试的数据
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2Vector('testDigits/%s' % fileNameStr)
        classifierResult=Classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if (classifierResult != classNumStr):
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
            errorCount+=1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

if __name__== '__main__':
    print("KNN training Started")
    group,labels = CreateDataSet()
    result = Classify0([0,0],group,labels,3)
    print("[0,0] is ",result)
    returnMat,classLabelVector = file2matrix('datingTestSet.txt')
    print(returnMat)
    print(classLabelVector)
    #使用matplotlib 创建散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:,1],returnMat[:,2],15.0*array(classLabelVector), 15.0*array(classLabelVector))
    ax.axis([-2, 25, -0.2, 2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    #plt.show()
    normMat, ranges, minVals = autoNorm(returnMat)
    datingClassTest()
    handwritingClassTest()