
from nltk.stem import PorterStemmer
import os, os.path
import math
import operator
from collections import OrderedDict

# This function returns the stop words list that needs to be elemenated
def stop_words():
    stop_words_list = []
    stop_word_file = open("stop_words.txt", "r")
    stop_words_lines = stop_word_file.readlines()
    for line in stop_words_lines:  
        stop_words_list.append(line.rstrip())
    return stop_words_list

# This function reads the documents and stores the words in a list
def read_document():

    dirListing = os.listdir('../Documents')
    documents = []
    stop_words_list = stop_words()
    ps = PorterStemmer()
    document_clean_words = {}
    # Assigns the key of the dict the name of the document 
    for item in dirListing:
        if ".txt" in item:
            documents.append(item)

    # Iterates over the documents and creates a list of clean words to the respective key of document
    for document in documents:

        document_words_list = []
        final_document_words_list = []
        
        open_document = open('../Documents/' + document, "r")
        document_lines = open_document.readlines()

        for line in document_lines:
            line = line.strip()
            words_in_line = line.split(" ")
            
            for word in words_in_line:
                word = word.lower()
                punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~/+-'''
                numbers = '1234567890'
                if len(word)>2:
                    if word[-1] not in punctuations and word[-1] not in numbers:
                        if word[0] in punctuations:
                            word = word.replace(word[1], "")
                            if word not in stop_words_list:
                                document_words_list.append(word)
                        elif word not in stop_words_list:     
                            document_words_list.append(word)
                    elif word[-1] in punctuations and word[0] not in numbers:
                        word = word.replace(word[-1], "")
                        document_words_list.append(word)
                    elif word[0] in punctuations and word[-1] in punctuations:
                        word = word.replace(word[0], "")
                        word = word.replace(word[-1], "")
                        if word[0] not in numbers:
                            document_words_list.append(word)
        
        for word in document_words_list:
            final_document_words_list.append(ps.stem(word))

        document_clean_words[document] = final_document_words_list
        
    return document_clean_words

# This function returns a list of documents name
def get_documents_name_list(doc_words_list):
    word_doc_dict = []
    for doc, value in doc_words_list.items():
        if doc not in word_doc_dict:
            word_doc_dict.append(doc)
    return word_doc_dict

# This function calculates tf for the words in respective documents
def tf(doc_words_list):

    word_doc_dict = {}

    for doc, value in doc_words_list.items():
        if doc not in word_doc_dict:
            word_doc_dict[doc] = {}

    for doc, value in word_doc_dict.items():
        for terms in doc_words_list[doc]:
            if terms not in word_doc_dict[doc]:
                word_doc_dict[doc][terms] = [1]
            elif terms in word_doc_dict[doc]:
                word_doc_dict[doc][terms][0] += 1
            
        for words in word_doc_dict[doc]:
            tf = word_doc_dict[doc][words][0]/len(doc_words_list[doc])
            word_doc_dict[doc][words][0] = tf

    return word_doc_dict

# This function calculate idf for the terms in respective documents
def idf(tf_dict, cleanwords_document_list):

    for doc, term in tf_dict.items():
        for words in term:
            for doc1, term1 in cleanwords_document_list.items():
                if words in cleanwords_document_list[doc1] and len(tf_dict[doc][words]) < 3:
                    tf_dict[doc][words].append(1)
                if words in cleanwords_document_list[doc1] and len(tf_dict[doc][words]) > 2: 
                    tf_dict[doc][words][1] += 1
    
    for doc, term in tf_dict.items():
        for words in term:
            idf = math.log10(len(tf_dict)/tf_dict[doc][words][1])
            tf_dict[doc][words][1] = idf

    return tf_dict

# This function calculates tf*idf for each word
def tf_idf(idf_document):

    for doc, terms in idf_document.items():
        for term in terms:
            tf_idf = idf_document[doc][term][0] * idf_document[doc][term][1]
            idf_document[doc][term] = tf_idf

    return idf_document

# This function returns a text document of topN words upon user's input
def topN_tf_idf(tf_idf_document, documents_name_list, topN):

    desc_tf_idf_document = []
    topN_words = []
    countDoc = 0
    countWords = 0
    f= open("topNwordInDoc.txt","w+")

    for doc, terms in tf_idf_document.items():
        desc_tf_idf_document.append(OrderedDict(sorted(terms.items(), key=lambda t: t[1], reverse=True))) 

    for items in desc_tf_idf_document:
        f.write(documents_name_list[countDoc] + "    ") 
        for values in items:
            if countWords != topN:
                f.write(values + ";")
            if countWords == topN:
                countWords = 0
                f.write(values + "\n")
                break
            countWords += 1
        countDoc += 1
    f.close()

    return "Top " + str(topN) + " words added in topNwordInDoc.txt"


cleanwords_document_list = read_document()

documents_name_list = get_documents_name_list(cleanwords_document_list)

tf_document = tf(cleanwords_document_list)

idf_document = idf(tf_document, cleanwords_document_list)

tf_idf_document = tf_idf(idf_document)

topN = int(input("Enter topN terms: "))
topN_keywords = topN_tf_idf(tf_idf_document, documents_name_list, topN) 







                
        
