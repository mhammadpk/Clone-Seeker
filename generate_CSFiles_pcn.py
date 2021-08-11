import javalang
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import re
from  more_itertools import unique_everseen
import os
from rake_nltk import Rake

def removeStopWords(queryTokens):
    filtered_tokens = [word for word in queryTokens if word not in stopwords.words('english')]
    return filtered_tokens


def removePrepositions(tokens):
    tagged = nltk.pos_tag(tokens)
    filtered_tokens = tagged[0:6]
    return filtered_tokens


def stemming(tokens):
    stemmer = SnowballStemmer(language='english')
    # stemmer = PorterStemmer()
    filtered_tokens = []
    for token in tokens:
        filtered_tokens.append(stemmer.stem(token))
    return filtered_tokens


def camel_case_split(tokens):
    filtered_tokens = []
    for i in tokens:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i)
        filtered_token = [m.group(0) for m in matches]
        filtered_tokens = filtered_tokens + filtered_token
    return filtered_tokens


def splitUnderscore(tokens):
    filtered_tokens = []

    for tok in tokens:
        matches = tok.split('_')
        filtered_tokens = filtered_tokens + matches
    return filtered_tokens


def getSynonyms(tokens):
    filtered_tokens = []
    for tok in tokens:
        for syn in wordnet.synsets(tok):
            for l in syn.lemmas():
                filtered_tokens.append(l.name())
    return filtered_tokens


def removeSingleCharacters(query):
    document =[w for w in query if len(w)>1]
    # (re.sub(r'\s+[a-zA-Z]\s+', ' ', ' '.join(query))).split(' ')
    return document


def convertLowerCase(tokens):
    lower = []
    for x in tokens:
        lower.append(x.lower())
    return lower

def getIdentifiers(tokenizeCode):
    queryTokens = []
    for m in range(len(tokenizeCode)):
        tokentype = tokenizeCode[m].__class__.__name__
        if (tokentype == "Null" or tokentype == "Keyword" or tokentype == "Separator" or tokentype == "Modifier" or
                tokentype == "Operator" or tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or
                tokentype == 'Integer' or tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or
                tokentype == 'HexInteger' or tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or
                tokentype == 'Boolean' or tokentype == 'Literal' or tokentype == 'Character' or tokentype == 'String'):
            continue
        else:
            queryTokens.append(tokenizeCode[m].value)
    return queryTokens

def getCodeQuery(tokenizeCode,docString):
    queryTokens = []
    for m in range(len(tokenizeCode)):
        tokentype = tokenizeCode[m].__class__.__name__
        # Remove reserve words, operators,seperators,
        # Modifier such as public, static
        # Keyword such as void
        # Seperator such as (, )
        ###Need to think for values for String value
        # Null values

        if (tokentype == "Null" or tokentype == "Keyword" or tokentype == "Separator" or tokentype == "Modifier" or
                tokentype == "Operator" or tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or
                tokentype == 'Integer' or tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or
                tokentype == 'HexInteger' or tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or
                tokentype == 'Boolean' or tokentype == 'Literal' or tokentype == 'Character' or tokentype == 'String'):
            continue
        else:
            queryTokens.append(tokenizeCode[m].value)

    # # Remove stop words
    docstringFilter2=nltk.word_tokenize(docString)
    #
    filter2 = removeStopWords(docstringFilter2)
    # # queryTokens
    # # Remove prepositions
    # # Perform stemming
    filterSC = removeSingleCharacters(filter2)

    filter3 = filterSC+queryTokens

    filter4 = filter3
    #removePrepositions(filter2)
    # Perform camel case
    filter5 = camel_case_split(filter4)
    # Perform underscore splittings
    filter6 = splitUnderscore(filter5)
    filter7=removeSingleCharacters(filter6)
    # Convert into lower case
    filter8 = convertLowerCase(filter7)
    # filter9 = stemming(filter8)

    # filter7 = getSynonyms(set(filter6))
    filter10 = list(unique_everseen(filter8))#set(filter9)
    return ' '.join(filter10)

def generateCloneDocument():
    ######Extracting clone method references belong to each functionality and each testing file#########
    df = pd.read_excel("C:\PhD\Autocomplete\Prediction code\JavaPrediction\Search_results\BCB_Data.xlsx",
                       sheet_name='Sheet3')
    array = df.as_matrix()
    totalClones = array.shape[0]
    totalNotSampleClones = 0
    funtionalClones = {}
    clones_ES = []
    # Generate table 4 on distributtion of clones by functions and Table
    # j = 0
    for i in range(0, totalClones):
        # funtionalClones[array[i][0]] = funtionalClones.get(array[i][0], 0) + 1
        clones_ES.append(array[i, :])
    #     j = j + 1
    #     reference = []
    #     if funtionalClones.get(array[i][0]) == None:
    #         reference.append(list(array[i, :]))
    #         funtionalClones[array[i][0]] = reference
    #     else:
    #         references = funtionalClones[array[i][0]]
    #         references.append(list(array[i, :]))
    #         funtionalClones[array[i][0]] = list(references)
    # uniqueFuntionalClones = {}
    #
    # for j in range(0,len(funtionalClones)):
    #     uniqueFuntionalClones[list(funtionalClones)[j]]=np.array(funtionalClones[list(funtionalClones)[j]])#np.unique(np.array(funtionalClones[list(funtionalClones)[j]]), axis=0)

    clonedCode = []
    # FUNCTIONALITY_ID  	DOC_NAME  	DESCRIPTION  	TYPE  	NAME  	STARTLINE  	ENDLINE  
    # FUNTIONID  	FUNCTIONALITY_ID  	NAME  	DESCRIPTION  	TYPE  	NAME  	STARTLINE  	ENDLINE  

    cloneCodeFunction = []
    docStringFunc = []
    functionIDFunc = []
    functionTypeID = []
    #  for oind in range(0,len(uniqueFuntionalClones)):
    # clones_ES=uniqueFuntionalClones[list(uniqueFuntionalClones)[oind]]
    #     cloneCodeList = []
    #    docStrings = []
    #   functionId = []
    for ind in range(0, len(clones_ES)):
        #############Extract clone code##############
        # for ind in range(len(clones_ES)):
        #############need to uncomment
        path = "C:/PhD/Autocomplete/Prediction code/JavaPrediction/bcb_reduced-Search"
        filepath = path + "/" + str(clones_ES[ind][1]) + "/" + clones_ES[ind][4] + "/" + clones_ES[ind][5]
        with open(filepath, encoding="utf8", errors='ignore') as f:
            content = f.readlines()
        cloneLines = content[int(clones_ES[ind][6]) - 1: int(clones_ES[ind][7])]
        clonedCode.append(cloneLines)
        docString = clones_ES[ind][3]
        tokenizeCode = list(javalang.tokenizer.tokenize(''.join(cloneLines)))
        orignalTokens = []
        for i in range(0, len(tokenizeCode)):
            orignalTokens.append(tokenizeCode[i].value)

        codeQuery = getCodeQuery(tokenizeCode, "")
        ############need to uncomment

        # docStrings.append(codeQuery)
        # cloneCodeList.append(" ".join(orignalTokens))
        # functionId.append(clones_ES[ind][0])

        # cloneCodeFunction.append(" ".join(orignalTokens))#cloneCodeFunction+cloneCodeList
        docStringFunc.append(codeQuery)  # docStringFunc + docStrings
        # functionIDFunc.append(str(clones_ES[ind][0]))#functionIDFunc+functionId
        functionTypeID.append(str(clones_ES[ind][1]))

    # total_inputs =list(zip(cloneCodeFunction, docStringFunc)) #list(unique_everseen(list(zip(cloneCodeFunction, docStringFunc))))
    # inputCloneCode = []
    # inputdocStrings = []
    # inputCloneCode[:], inputdocStrings[:] = zip(*total_inputs)

    # with open("cloneCode_STEM.txt", "w", encoding="utf-8") as outfile:
    #     outfile.write("<CODE_SPLIT>".join(cloneCodeFunction))

    with open("docString_WU_W.txt", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(docStringFunc))

    # with open("functionID_W_1.txt", "w", encoding="utf-8") as outfile:
    #         outfile.write("\n".join(functionIDFunc))

    # with open("functionTypeID.txt", "w", encoding="utf-8") as outfile:
    #     outfile.write("\n".join(functionTypeID))

    # cloneCodeFunction[list(uniqueFuntionalClones)[oind]]=cloneCodeList
    # docStringFunc[list(uniqueFuntionalClones)[oind]] = docStrings


def readFile(filename):
    text_file = open(filename, 'r', encoding="utf8")
    data = text_file.read()
    listOfSentences = data.split("\n")
    return listOfSentences

def buildSearchCorpus(path, selectedProblems):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(path)] for
             val in
             sublist]
    for i in range(len(files)):
        try:
            # tokens = []
            filename = files[i].split('\\')[6]
            foldername = files[i].split('\\')[5]
            if foldername in selectedProblems:
                with open(files[i], encoding="utf8", errors='ignore') as f:
                    content = f.readlines()
                orignalcode="\n".join(content)
                # content[0] = "socCLNE " + content[0]
                # content[len(content)-1] = content[len(content)-1]+" eocCLNE"

                tokenize = list(javalang.tokenizer.tokenize(''.join(content)))

                # for m in range(len(tokenize)):
                #     tokentype = tokenize[m].__class__.__name__
                #     if (tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or tokentype == 'Integer' or
                #             tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or tokentype == 'HexInteger' or
                #             tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or tokentype == 'Boolean' or tokentype == 'Literal'):
                #         tokens.append("<num_val>")
                #     elif (tokentype == 'Character' or tokentype == 'String'):
                #         tokens.append("<str_val>")
                #     else:
                #         tokens.append(tokenize[m].value)


                codeQuery = getCodeQuery(tokenize, "")

                with open("Data/baseline_pcn.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
                    txt_file.write(codeQuery + "\n")

                # with open("Data/clonecorpus_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
                #     txt_file.write("<soc> "+" ".join(tokens)+" <eoc>\n")

                with open("Data/orignalcode_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
                    txt_file.write(orignalcode)
                    txt_file.write(" <CODESPLIT> ")

                with open("Data/functionalityId_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
                    txt_file.write(foldername+"\n")

                with open("Data/fileName_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
                    txt_file.write(filename+"\n")

        except Exception as e:
            print(e)
            # with open("Data/clonecorpus_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
            #     txt_file.write("<EXCEPT>\n")
            #
            # with open("Data/functionalityId_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
            #     txt_file.write(foldername + "\n")
            #
            # with open("Data/fileName_projectcodenet.txt", "a", encoding="utf-8", errors='ignore') as txt_file:
            #     txt_file.write(filename + "\n")
            print(foldername+"/"+filename)
            pass
def extractKeywords(document):
    r = Rake()
    r.extract_keywords_from_text(" ".join(document))
    # To get keyword phrases ranked highest to lowest with scores.
    items=r.get_word_frequency_distribution()#r.get_word_degrees()
    top_n=10
    topn_values=sorted(items, key=items.get, reverse=True)[:top_n]
    return topn_values

def createAutomaticCorpus():
    clonecode = readFile("Data/baseline_pcn.txt")
    functionId = readFile("Data/fileName_projectcodenet.txt")
    functionTypeId=readFile("Data/functionalityId_projectcodenet.txt")
    sFid=set(functionTypeId)
    # distinctFId = [int(x) for x in sFid]
    # distinctFId.sort()
    with open("Data/automatic_pcn_10.txt", "a", encoding="utf-8") as outfile:
        for funcTypeId in sFid:
            # filteredList=zip(*((id, other) for id, other in zip(functionTypeId, clonecode) if id ==funcTypeId))
            cloneFilterCode=[other for id, functionId,other in zip(functionTypeId, functionId, clonecode) if str(id) == str(funcTypeId)]
            functionFilterId = [functionId for id, functionId, other in zip(functionTypeId, functionId, clonecode) if
                               str(id) == str(funcTypeId)]
            #cloneFilterCode=list(get_col(filteredList, 1))

            summarizedText=extractKeywords(cloneFilterCode)
            for i in range(0,len(cloneFilterCode)):# print(str(labels_genie[index]) + ":" + str(sentence))
                    outfile.write(
                        str(funcTypeId) + "<SPLIT>" + functionFilterId[i]+"<SPLIT>"+' '.join(summarizedText) +
                        "<SPLIT>" + cloneFilterCode[i] + "\n")



# selectedProblems=readFile("Data\genericProblems.txt")
# buildSearchCorpus("C:\PhD\Search Guide Model Papers\Clone-Advisor-main\Project_CodeNet_Java250",selectedProblems)
createAutomaticCorpus()