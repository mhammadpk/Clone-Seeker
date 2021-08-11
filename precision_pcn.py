from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import torch
import math
import numpy as np
from statistics import mean
import javalang


def calculateFunctionalMetrics(gtFuncationalId, topKFuncationalId):
    rank = 1.0
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0
    top10 = 0.0
    for j in range(10):
        if gtFuncationalId.strip() == topKFuncationalId[j].strip():
            if rank <= 1.0:
                top1 += 1.0
            if rank <= 3.0:
                top3 += 1.0
            if rank <= 5.0:
                top5 += 1.0
            if rank <= 10.0:
                top10 += 1.0
        rank = rank + 1.0
    return top1 / 1.0, top3 / 3.0, top5 / 5.0, top10 / 10.0


def getFunctionalGroup(cloneList, total_function_snippets):
    functionIds = []
    snippets = [x[1] for x in total_function_snippets]
    indexF = -1
    functions = [x[0] for x in total_function_snippets]
    for i in range(0, len(cloneList)):
        for j in range(0, len(snippets)):
            if snippets[j] == cloneList[i]:
                indexF = j
                break
        if indexF == -1:
            print(i)
        else:
            functionIds.append(functions[indexF])

    return functionIds


def readFile(filename):
    text_file = open(filename, 'r', encoding="utf8")
    data = text_file.read()
    listOfSentences = data.split("\n")
    return listOfSentences


def getJavaTokenizeList(predictedClones):
    cloneList = []
    for i in range(0, len(predictedClones)):
        text = predictedClones[i]
        text2=text.split(' ')
        text3=" ".join(text2[1:len(text2)-1])
        text4=text3.replace("<num_val>","NUMVAL")
        text5=text4.replace("<str_val>", "STRVAL")
        text6=text5.replace("<soc>","")
        # text2 = text[: text.find('<soc>') if '<soc>' else None]
        # text3 = text2[: text2.find('<eoc>') if '<eoc>' else None]
        tokenizeCode = list(javalang.tokenizer.tokenize(text6))
        tokens = []
        tokens.append("<soc>")
        for m in range(0, len(tokenizeCode)):
            tokenval = tokenizeCode[m].value
            if (tokenval == 'NUMVAL'):
                tokens.append("<num_val>")
            elif (tokenval == 'STRVAL'):
                tokens.append("<str_val>")
            else:
                tokens.append(tokenizeCode[m].value)
        tokens.append("<eoc>")
        cloneList.append(" ".join(tokens))

    return cloneList


def cleanSOC():
    generatedclones = []
    predictedClones2 = readFile('finaldataset/generatedClones_pcn_20token_test.txt')
    for i in range(0, len(predictedClones2)):
        text = predictedClones2[i]
        text2 = text.split(' ')
        text3 = " ".join(text2[1:len(text2)])
        text6 = text3.replace(" <soc> ", " ")
        generatedclones.append("<soc> " + text6)

    for j in range(0, len(generatedclones)):
        with open("finaldataset/generatedclones_clean.txt", "a", encoding="utf-8") as outfile:
            outfile.write(str(generatedclones[j]) + "\n")


def readData():
    text_file = open("ExpData/generatedClones_bm.txt", 'r')
    data = text_file.read()
    listOfSentences = data.split("\n")
    indexes = []
    for i in range(len(listOfSentences)):
        if listOfSentences[i] == "":
            indexes.append(i)

    predictedClones = [e for i, e in enumerate(listOfSentences) if
                       i not in indexes]

    text_file = open("Data/benchmarkmethods.txt", 'r')
    data = text_file.read()
    listOfSentences = data.split("\n")
    orignalClones = [e for i, e in enumerate(listOfSentences) if
                     i not in indexes]

    text_file = open("Data/functionalityIds_bm.txt", 'r')
    data = text_file.read()
    listOfSentences = data.split("\n")
    functionalityIdBm = [e for i, e in enumerate(listOfSentences) if
                     i not in indexes]


    text_file = open("Data/inputSequence_bm.txt", 'r')
    data = text_file.read()
    listOfSentences = data.split("\n")
    inputSequences = [e for i, e in enumerate(listOfSentences) if
                      i not in indexes]

    return predictedClones, orignalClones, inputSequences, functionalityIdBm

def getTopKCloneResultsDocSim(cloneLibrary, clone, k):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([clone] + cloneLibrary)
    # Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents
    cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes
    total_score_snippets = [(score, title) for score, title in zip(document_scores, cloneLibrary)]
    topk = sorted(total_score_snippets, reverse=True, key=lambda x: x[0])[:k]
    return topk

def readFile(filename):
    text_file = open(filename, 'r', encoding="utf8")
    data = text_file.read()
    listOfSentences = data.split("\n")
    return listOfSentences
def get_col(arr, col):
    return map(lambda x: x[col], arr)
def functionalEvaluation():
    cloneCorpus = readFile("Data/baseline_pcn.txt")
    cloneCode = list(set(cloneCorpus))
    functionalityIds = readFile("Data/functionalityId_projectcodenet.txt")  # np.load("functionalityIds.npy")
    total_function_snippets = [(s, t) for s, t in zip(functionalityIds, cloneCorpus)]
    # cloneOutput, orignalOutput, inputsequences, functionalityIdBm = readData()
    mrrList = []
    top1List = []
    top3List = []
    top5List = []
    top10List = []
    foldername = "Results/"
    it=0
    for m in range(0, len(functionalityIds)):#len(cloneOutput)):
        ###get ground truth functional Id
        gtFuncationalId = functionalityIds[m]#getFunctionalGroup([orignalOutput[m]], total_function_snippets)
      # if gtFuncationalId in ['40','5','9']:#,'9']:#['10','21','25','37','40','43','5']:
          #['10','17','18','21','22','25','32','33','34','37','39','40','43','5','9']:#['5']:#"['10', '22', '25', '26']:
        topKPredicted = getTopKCloneResultsDocSim(cloneCode, cloneCorpus[it], 10)
        topKFuncationalId = getFunctionalGroup(list(get_col(topKPredicted, 1)), total_function_snippets)
        cloneOutputList = []
        cloneOutputList.append(cloneCorpus[it])
        top1, top3, top5, top10 = calculateFunctionalMetrics(gtFuncationalId, topKFuncationalId)
        top1List.append(top1)
        top3List.append(top3)
        top5List.append(top5)
        top10List.append(top10)
        it=it+1
        with open(foldername + "precision_pcn.txt", "a", encoding="utf-8") as outfile:
            outfile.write(str(top1) + ',' + str(top3) + ',' + str(top5) + ',' + str(top10))
            outfile.write('\n')
            outfile.close()

    print("P1", str(mean(top1List)))
    print("P3", str(mean(top3List)))
    print("P5", str(mean(top5List)))
    print("P10", str(mean(top10List)))

functionalEvaluation()