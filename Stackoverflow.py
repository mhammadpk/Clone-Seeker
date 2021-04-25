import mathfrom  more_itertools import unique_everseenimport randomfrom sklearn.feature_extraction.text import TfidfVectorizerimport javalangimport pandas as pdfrom nltk.corpus import stopwordsimport nltkfrom nltk.stem.snowball import SnowballStemmerimport refrom nltk.corpus import wordnetimport osimport numpy as npfrom os.path import isfile, joinfrom collections import Counter, defaultdictdef removeStopWords(queryTokens):    filtered_tokens = [word for word in queryTokens if word not in stopwords.words('english')]    return filtered_tokensdef stemming(tokens):    stemmer = SnowballStemmer(language='english')    # stemmer = PorterStemmer()    filtered_tokens = []    for token in tokens:        filtered_tokens.append(stemmer.stem(token))    return filtered_tokensdef camel_case_split(tokens):    filtered_tokens = []    for i in tokens:        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i)        filtered_token = [m.group(0) for m in matches]        filtered_tokens = filtered_tokens + filtered_token    return filtered_tokensdef splitUnderscore(tokens):    filtered_tokens = []    for tok in tokens:        matches = tok.split('_')        filtered_tokens = filtered_tokens + matches    return filtered_tokensdef removeSingleCharacters(query):    document =[w for w in query if len(w)>1]    # (re.sub(r'\s+[a-zA-Z]\s+', ' ', ' '.join(query))).split(' ')    return documentdef convertLowerCase(tokens):    lower = []    for x in tokens:        lower.append(x.lower())    return lowerdef normalizeStackOverflowQuery(docString):    queryTokens = []    # # Remove stop words    docstringFilter2=nltk.word_tokenize(docString)    filter2 = removeStopWords(docstringFilter2)    filter4 = filter2    # Perform camel case    filter5 = camel_case_split(filter4)    # Perform underscore splittings    filter6 = splitUnderscore(filter5)    filter7=removeSingleCharacters(filter6)    # Convert into lower case    filter8 = convertLowerCase(filter7)    filter9=stemming(filter8)    filter10 = list(unique_everseen(filter9))    return ' '.join(filter10)def identfiyFunctionalityTypeId(topKDocString, corpus,functionalityType,functionalityTypeList):    topKFunctionalityTypeId=[]    for i in range(0,len(topKDocString)):       filteredIndexes=[x for x in range(0,len(corpus)) if corpus[x]==topKDocString[i]]       functionalityTypeIndexes=[functionalityTypeList[filteredIndexes[j]] for j in range(0,len(filteredIndexes))]       status=[x == functionalityType for x in functionalityTypeIndexes]       check=False       for x in status:           if x == True:               check=True               break       if check==True:           topKFunctionalityTypeId.append(functionalityType)       else:           topKFunctionalityTypeId.append(-1)    return topKFunctionalityTypeIddef readClusterData():    text_file = open("codeQueryResults_AA_STEM_10.txt", 'r', encoding="utf8")#open("codeQueryResults_AA_STEM_15.txt", 'r', encoding="utf8")    data = text_file.read()    listOfSentences = data.split("\n")    functionalityType = []    docString = []    summary = []    for i in range(0, len(listOfSentences)):        sentenceParts = listOfSentences[i].split('<SPLIT>')        if (len(sentenceParts) == 4):            functionalityType.append(sentenceParts[0])            summary.append(sentenceParts[2])            docString.append(" ".join(list(unique_everseen((sentenceParts[2]+" "+sentenceParts[3]).split(" ")))))    return functionalityType, docString    def calculateStackOverflowStatistics():    queries = ['How to download and save a file from the Internet using Java?', #2               'How can I generate an MD5 hash?',                               #3               'Error while copying files from source to destination java',     #4               'Java decompress archive file',                                  #5               'Java: Accessing a File from an FTP Server',                     #6               'Basic Bubble Sort with ArrayList in Java',                      #7               'GEF editor functionality to view',                              #8               'ScrollingGraphicalViewer Select and Unselect listener',         #9               'Java: Rollback Database updates?',                              #10               'Creating a Eclipse Java Project from another project, programatically', #11               'Java Display the Prime Factorization of a number',                      #12               'Random shuffling of an array',                                          #13               'First occurrence in a binary search',                                   #14               'Java - How to Load a Custom Font From a Resources Folder',              #15               'Issues in RSA encryption in Java class',                                #17               'How can I play sound in Java?',                                         #18               'Is there a way to take a screenshot using Java and save it to some sort of image?', #19               'printing the results of a fibbonacci series',                                                 #20               'implementing GAE XMPP service as an external component to an existing XMPP server (e.g. ejabberd or OpenFire)', #21               'How to Encrypt/Decrypt text in a file in Java',                                     #22               'Resize an Array while keeping current elements in Java?',                           #23               'How to open the default webbrowser using java',                                     #24               'Open a file using Desktop(java.awt)',                                               #25               'Java: get greatest common divisor',                                                 #26               'How to invoke a method in java using reflection',                                   #27               'Parsing XML file with DOM (Java)',                                                  #28               'Change date format in a Java string',                                               #29               'How to create a zip file in Java',                                                  #30               'Is it possible to select multiple directories at once with JFileChooser',           #31               'How can I send an email by Java application using GMail, Yahoo, or Hotmail?',       #32               'File containing its own checksum',                                                  #33               'Launching external process from Java : stdout and stderr',                          #34               'With Java reflection how to instantiate a new object, then call a method on it?',   #35               'Create MySQL database from Java',                                                   #36               'Reading a binary input stream into a single byte array in Java',                    #37               'Get MAC address on local machine with Java',                                        #38               'How to delete a folder with files using Java',                                      #39               'Fastest way to read a CSV file java',                                               #40               'Transposing a matrix from a 2D array',                                              #41               'Using Regular Expressions to Extract a Value in Java',                              #42               'How to copy an entire content from a directory to another in Java?',                #43               'Check string for palindrome',                                                       #44               'java pdf file write']                                                              #45               # 'How to Write PDF using Java']    #    functionalityType = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15','17','18','19','20','21','22','23','24',                         '25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45']    #cloneCode = readFile("cloneCode.txt")        #For manual search corpus    docString = readFile("docString_M_STEM.txt")        #For without annotation search corpus    #docString = readFile("docString_W_STEM.txt")            functionalityTypeId=readFile("functionTypeID.txt")        #functionIds=readFile("functionID.txt")            #functionalityTypeId, docString=readClusterData()        index = 10    print("ID & MRR & P@1 & P@5 & P@10")    for i in range(0, len(queries)):        mrr = 0.0        top1 = 0.0        top3 = 0.0        top5 = 0.0        top10 = 0.0        normalizedQuery = normalizeStackOverflowQuery(queries[i])        topKDocList = semanticSearch(docString, normalizedQuery, 10, 1)        topKDocString=list(get_col(topKDocList, 1))        topKfunctionTypeID = identfiyFunctionalityTypeId(topKDocString, docString,functionalityType[i],functionalityTypeId)        rank = 1.0        for j in range(10):            if topKfunctionTypeID[j] == functionalityType[i]:                mrr += 1.0 / rank                if rank <= 1.0:                    top1 += 1.0                if rank <= 3.0:                    top3 += 1.0                if rank <= 5.0:                    top5 += 1.0                if rank <= 10.0:                    top10 += 1.0                break            rank = rank + 1.0        #Hit rate        hit1 = 0.0        hit3 = 0.0        hit5 = 0.0        hit10 = 0.0        rank = 1.0        for j in range(10):            if topKfunctionTypeID[j] == functionalityType[i]:                if rank <= 1.0:                    hit1 += 1.0                if rank <= 3.0:                    hit3 += 1.0                if rank <= 5.0:                    hit5 += 1.0                if rank <= 10.0:                    hit10 += 1.0            rank = rank + 1.0        print(str(functionalityType[i]) + ' & '+ str(mrr) + ' & '+ str(hit1 / 1.0)+' & '+ str(hit5 / 5.0)+' & '+str(hit10 / 10.0))def semanticSearch(cloneLibrary, clone, k, isSemantic):    if isSemantic == 1:        from sklearn.feature_extraction.text import TfidfVectorizer        from sklearn.metrics.pairwise import linear_kernel        vectorizer = TfidfVectorizer()        vectors = vectorizer.fit_transform([clone] + cloneLibrary)        # Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents        cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()        document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes        total_score_snippets = [(score, title) for score, title in zip(document_scores, cloneLibrary)]        topk = sorted(total_score_snippets, reverse=True, key=lambda x: x[0])[:k]        return topkdef readFile(filename):    text_file = open(filename, 'r',encoding="utf8")    data = text_file.read()    listOfSentences = data.split("\n")    return listOfSentencesdef get_col(arr, col):    return map(lambda x: x[col], arr)calculateStackOverflowStatistics()