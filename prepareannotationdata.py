import mathfrom  more_itertools import unique_everseenimport randomfrom sklearn.feature_extraction.text import TfidfVectorizerimport javalangimport pandas as pdfrom nltk.corpus import stopwordsimport nltkfrom nltk.stem.snowball import SnowballStemmerimport refrom nltk.corpus import wordnetfrom sklearn.metrics import ndcg_scoreimport jsonfrom statistics import meanfrom statistics import variancefrom statistics import stdevimport osfrom shutil import copyfileimport filecmpimport numpy as npfrom os import listdirfrom os.path import isfile, joinfrom nltk.translate.bleu_score import SmoothingFunctiondef removeStopWords(queryTokens):    filtered_tokens = [word for word in queryTokens if word not in stopwords.words('english')]    return filtered_tokensdef removePrepositions(tokens):    tagged = nltk.pos_tag(tokens)    filtered_tokens = tagged[0:6]    return filtered_tokensdef stemming(tokens):    stemmer = SnowballStemmer(language='english')    filtered_tokens = []    for token in tokens:        filtered_tokens.append(stemmer.stem(token))    return filtered_tokensdef camel_case_split(tokens):    filtered_tokens = []    for i in tokens:        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i)        filtered_token = [m.group(0) for m in matches]        filtered_tokens = filtered_tokens + filtered_token    return filtered_tokensdef splitUnderscore(tokens):    filtered_tokens = []    for tok in tokens:        matches = tok.split('_')        filtered_tokens = filtered_tokens + matches    return filtered_tokensdef removeSingleCharacters(query):    document =[w for w in query if len(w)>1]    # (re.sub(r'\s+[a-zA-Z]\s+', ' ', ' '.join(query))).split(' ')    return documentdef convertLowerCase(tokens):    lower = []    for x in tokens:        lower.append(x.lower())    return lowerdef getCodeQuery(tokenizeCode,docString):    queryTokens = []    for m in range(len(tokenizeCode)):        tokentype = tokenizeCode[m].__class__.__name__        if (tokentype == "Null" or tokentype == "Keyword" or tokentype == "Separator" or tokentype == "Modifier" or                tokentype == "Operator" or tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or                tokentype == 'Integer' or tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or                tokentype == 'HexInteger' or tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or                tokentype == 'Boolean' or tokentype == 'Literal' or tokentype == 'Character' or tokentype == 'String'):            continue        else:            queryTokens.append(tokenizeCode[m].value)    # # Remove stop words    docstringFilter2=nltk.word_tokenize(docString)    #    filter2 = removeStopWords(docstringFilter2)    filterSC = removeSingleCharacters(filter2)    filter3 = filterSC+queryTokens    filter4 = filter3    # Perform camel case    filter5 = camel_case_split(filter4)    # Perform underscore splittings    filter6 = splitUnderscore(filter5)    filter7=removeSingleCharacters(filter6)    # Convert into lower case    filter8 = convertLowerCase(filter7)    filter9 = stemming(filter8)    filter10 = list(unique_everseen(filter9))#set(filter9)    return ' '.join(filter10)def generateCloneDocument():    ######Extracting clone method references belong to each functionality and each testing file#########    df = pd.read_excel("C:\PhD\Autocomplete\Prediction code\JavaPrediction\Search_results\BCB_Data.xlsx",                       sheet_name='Sheet3')    array = df.as_matrix()    totalClones = array.shape[0]    totalNotSampleClones = 0    funtionalClones = {}    clones_ES = []    for i in range(0,totalClones):        clones_ES.append(array[i, :])    clonedCode = []    cloneCodeFunction=[]    docStringFunc=[]    functionIDFunc=[]    functionTypeID= []    for ind in range(0,len(clones_ES)):            #############Extract clone code##############            path = "C:/PhD/Autocomplete/Prediction code/JavaPrediction/bcb_reduced-Search"            filepath = path + "/" + str(clones_ES[ind][1]) + "/" + clones_ES[ind][4] + "/" + clones_ES[ind][5]            with open(filepath, encoding="utf8", errors='ignore') as f:                content = f.readlines()            cloneLines = content[int(clones_ES[ind][6]) - 1: int(clones_ES[ind][7])]            clonedCode.append(cloneLines)            docString = clones_ES[ind][3]            tokenizeCode = list(javalang.tokenizer.tokenize(''.join(cloneLines)))            orignalTokens = []            for i in range(0,len(tokenizeCode)):                orignalTokens.append(tokenizeCode[i].value)            codeQuery = getCodeQuery(tokenizeCode,"")            #codeQuery = getCodeQuery(tokenizeCode,docString) #for manual annotation            ############need to uncomment            docStringFunc.append(codeQuery) #docStringFunc + docStrings            functionIDFunc.append(str(clones_ES[ind][0]))#functionIDFunc+functionId            functionTypeID.append(str(clones_ES[ind][1]))    #total_inputs =list(zip(cloneCodeFunction, docStringFunc)) #list(unique_everseen(list(zip(cloneCodeFunction, docStringFunc))))    #inputCloneCode = []    #inputdocStrings = []    #inputCloneCode[:], inputdocStrings[:] = zip(*total_inputs)    # with open("cloneCode_STEM.txt", "w", encoding="utf-8") as outfile:    #     outfile.write("<CODE_SPLIT>".join(cloneCodeFunction))    with open("docString_W_S.txt", "w", encoding="utf-8") as outfile:            outfile.write("\n".join(docStringFunc))    # with open("functionID.txt", "w", encoding="utf-8") as outfile:    #         outfile.write("\n".join(functionIDFunc))    #    # with open("functionTypeID.txt", "w", encoding="utf-8") as outfile:    #         outfile.write("\n".join(functionTypeID))def readFile(filename):    text_file = open(filename, 'r',encoding="utf8")    data = text_file.read()    listOfSentences = data.split("\n")    return listOfSentencesgenerateCloneDocument()