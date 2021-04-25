from collections import defaultdict
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from  more_itertools import unique_everseen

def get_col(arr, col):
	return map(lambda x: x[col], arr)


def readFile(filename):
	text_file = open(filename, 'r',encoding="utf8")
	data = text_file.read()
	listOfSentences = data.split("\n")
	return listOfSentences

def semanticSearch(cloneLibrary, clone, k, isSemantic):
	if isSemantic == 1:
		vectorizer = TfidfVectorizer()
		vectors = vectorizer.fit_transform([clone] + cloneLibrary)
		# Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents
		cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
		document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes
		total_score_snippets = [(score, title) for score, title in zip(document_scores, cloneLibrary)]
		if len(total_score_snippets)<900:
			k=len(total_score_snippets)
		topk = sorted(total_score_snippets, reverse=True, key=lambda x: x[0])[:k]
		return topk


def readBigCloneBenchDataset():
	text_file = open("BCB/type1.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	type1 = data.split("\n")

	text_file = open("BCB/type2.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	type2 = data.split("\n")

	text_file = open("BCB/vst3.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	vst3 = data.split("\n")

	text_file = open("BCB/st3.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	st3 = data.split("\n")

	text_file = open("BCB/mt3.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	mt3 = data.split("\n")

	text_file = open("BCB/wt4_1.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	wt41 = data.split("\n")

	text_file = open("BCB/wt4_2.txt", 'r', encoding="windows-1252")
	data = text_file.read()
	wt42 = data.split("\n")
	wt4=wt41+wt42

	return type1,type2,vst3,st3,mt3,wt4

def getMatchedIndexes(list1,list2):
	d = defaultdict(list)
	for index, item in enumerate(list2):
		d[item].append(index)
	matchedIndexes = [index2 for item2 in list1 for index2 in d[item2] if item2 in d]
	return matchedIndexes


def getFunctionIds(topKDocString,functionIds,docStrings):
	topKFunctionIds=[]
	for i in range(0,len(topKDocString)):
		for j in range(0,len(docStrings)):
			if topKDocString[i]==docStrings[j]:
				topKFunctionIds.append(functionIds[j])
				# break

	return list(unique_everseen(topKFunctionIds))[0:900]

def getRecall(topKFunctionID,functionId,functionId1,functionId2):
	###########Get list of true positive clone methods of particular function
	matched1=[functionId+"-"+functionId2[x] for x in range(0, len(functionId1)) if functionId1[x] == functionId]
	matched2=[functionId1[x]+"-"+functionId for x in range(0, len(functionId2)) if functionId2[x] == functionId]

	########Retrieve list of top-900 Clone-Seeker clone pairs
	list1 = [functionId+"-"+topKFunctionID[x] for x in range(len(topKFunctionID))]
	list2 = [topKFunctionID[x]+"-"+functionId for x in range(len(topKFunctionID))]

	########Get list of items which are common between top-900 and true positive clone methods
	m1=list(set(list1).intersection(matched1))
	m2 = list(set(list1).intersection(matched2))
	m3 = list(set(list2).intersection(matched1))
	m4 = list(set(list2).intersection(matched2))
	commonList=list(set(m1+m2+m3+m4))

	z=0

	try:
		z = (len(commonList))/(len(list(set(list(set(matched1))+list(set(matched2))))))
	except ZeroDivisionError:
		z = -1
	return z

def writeToFile(places,filename):
	with open(filename, 'w') as filehandle:
		for listitem in places:
			filehandle.write('%s\n' % listitem)
def get_col(arr, col):
	return map(lambda x: x[col], arr)

def getFunctionId(categoryClone):
	functionId1 = []
	functionId2 = []
	for i in range(1, len(categoryClone)):
		parts = categoryClone[i].split(',')
		functionId1.append(parts[0])
		functionId2.append(parts[1])
	return functionId1,functionId2


def readBCBCloneFile(filename):
	text_file = open(filename, 'r',encoding="utf8")
	data = text_file.read()
	listOfSentences = data.split("\n")
	functionIds=[]
	searchCorpus=[]
	docStrings=[]
	for s in listOfSentences:
		data=s.split('<SPLIT>')
		functionIds.append(data[1])
		docStrings.append(data[3])
		searchCorpus.append(" ".join(list(unique_everseen((data[2]+" "+data[3]).split(' ')))))

	return functionIds,searchCorpus,docStrings

def calculaterecall():

	#cloneCodes=readFile("cloneCode.txt")
	# searchCorpus=readFile("docString.txt")
	# docStrings=readFile("docString_W.txt")

	#print("Without Annotation,Stemming,Unique,Limit 900--New")
	searchCorpus=readFile("docString_W_STEM.txt")
	docStrings=readFile("docString_W_STEM.txt")
    ####For manual annotation
    #searchCorpus=readFile("docString_M_STEM.txt")
	#docStrings=readFile("docString_W_STEM.txt")
    
    
    ####For automatic annotation
    #functionIds,searchCorpus,docStrings=readBCBCloneFile('codeQueryResults_AA_STEM_10.txt')

    
	searchCorpusU = list(set(searchCorpus))
	docStringsU = list(set(docStrings))

	functionIds = readFile("functionID.txt")
	type1, type2, vst3, st3, mt3, wt4=readBigCloneBenchDataset()
	type1FuncId1,type1FuncId2=getFunctionId(type1)
	type2FuncId1, type2FuncId2 = getFunctionId(type2)
	vst3FuncId1, vst3FuncId2 = getFunctionId(vst3)
	st3FuncId1, st3FuncId2 = getFunctionId(st3)
	mt3FuncId1, mt3FuncId2 = getFunctionId(mt3)
	wt4FuncId1, wt4FuncId2 = getFunctionId(wt4)

	type1recall=[]
	type2recall = []
	vst3recall=[]
	st3recall=[]
	mt3recall=[]
	wt4recall=[]
	for x in range(0,len(docStringsU)):
			topKDocStringInfo=semanticSearch(searchCorpusU,docStringsU[x],900,1)
			topKDocString=list(get_col(topKDocStringInfo,1))
			selectedFunctionId=[m for m in range(0,len(docStrings)) if docStringsU[x]==docStrings[m]]

			topKFunctionID=getFunctionIds(topKDocString,functionIds,searchCorpus)
			type1recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], type1FuncId1, type1FuncId2))
			type2recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], type2FuncId1, type2FuncId2))
			vst3recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], vst3FuncId1, vst3FuncId2))
			st3recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], st3FuncId1, st3FuncId2))

			mt3recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], mt3FuncId1, mt3FuncId2))
			wt4recall.append(getRecall(topKFunctionID, functionIds[selectedFunctionId[0]], wt4FuncId1, wt4FuncId2))

	print("Type 1 Recall:"+str(mean([m for m in type1recall if m!=-1.0])))
	print("Type 2 Recall:" + str(mean([m for m in type2recall if m!=-1.0])))
	print("VST3 Recall:" + str(mean([m for m in vst3recall if m!=-1.0])))
	print("ST3 Recall:" + str(mean([m for m in st3recall if m!=-1.0])))
	print("MT3 Recall:" + str(mean([m for m in mt3recall if m!=-1.0])))
	print("WT4 Recall:" + str(mean([m for m in wt4recall if m!=-1.0])))

calculaterecall()