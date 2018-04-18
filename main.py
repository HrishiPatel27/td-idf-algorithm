
from nltk.stem import PorterStemmer
import os, os.path
import math
import time
import operator
from collections import OrderedDict

start_time = time.time()

def stop_words():
    stop_words_list = []
    stop_word_file = open("stop_words.txt", "r")
    stop_words_lines = stop_word_file.readlines()
    for line in stop_words_lines:  
        stop_words_list.append(line.rstrip())
    return stop_words_list

def read_document():
    dirListing = os.listdir('../Documents')
    documents = []
    stop_words_list = stop_words()
    for item in dirListing:
        if ".txt" in item:
            documents.append(item)
    document_clean_words = {}
    for document in documents:
        ps = PorterStemmer()
        document_words_list = []
        final_document_words_list = []
        open_document = open('../Documents/' + document, "r")
        document_lines = open_document.readlines()
        for line in document_lines:
            line = line.strip()
            words_in_line = line.split(" ")
            # stop_words_list = []
            for word in words_in_line:
                word = word.lower()
                punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~/+-'''
                numbers = '1234567890'
                if len(word)>2:
                    # print(word)
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
        
    # for key, value in document_clean_words.items():
    #     print(key, value)
    #     print("\n\n\n\n\n\n")
    return document_clean_words

def get_documents_name_list(doc_words_list):
    word_doc_dict = []
    for doc, value in doc_words_list.items():
        if doc not in word_doc_dict:
            word_doc_dict.append(doc)
    return word_doc_dict

def tf(doc_words_list):
    word_doc_dict = {}
    for doc, value in doc_words_list.items():
        if doc not in word_doc_dict:
            word_doc_dict[doc] = {}

    for doc, value in word_doc_dict.items():
        for terms in doc_words_list[doc]:
            # print(terms)
    
            if terms not in word_doc_dict[doc]:
                word_doc_dict[doc][terms] = [1]
            elif terms in word_doc_dict[doc]:
                word_doc_dict[doc][terms][0] += 1
            # elif terms in word_doc_dict[doc] and len(word_doc_dict[doc][terms]) == 0:
            #     word_doc_dict[doc][terms].append(2)
            # elif terms in word_doc_dict[doc] and len(word_doc_dict[doc][terms]) != 0:
            #     word_doc_dict[doc][terms][0] += 1
        for words in word_doc_dict[doc]:
            # if len(word_doc_dict[doc][words]) < 2:
            tf = word_doc_dict[doc][words][0]/len(doc_words_list[doc])
                # print(tf)
            word_doc_dict[doc][words][0] = tf
    # print(word_doc_dict)
    # for key, value in word_doc_dict.items():
    #     print(key, value)
    #     print("\n\n\n\n\n\n")

    return word_doc_dict

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
            # print(tf_dict[words])
            idf = math.log10(len(tf_dict)/tf_dict[doc][words][1])
            tf_dict[doc][words][1] = idf

    # print(tf_dict)
    # for key, value in tf_dict.items():
    #     print(key, value)
    #     print("\n\n\n\n\n\n")
    return tf_dict

def tf_idf(idf_document):

    for doc, terms in idf_document.items():
        for term in terms:
            tf_idf = idf_document[doc][term][0] * idf_document[doc][term][1]
            idf_document[doc][term] = tf_idf
    return idf_document

def topN_tf_idf(tf_idf_document, documents_name_list, topN):

    desc_tf_idf_document = []
    topN_words = []
    for doc, terms in tf_idf_document.items():
        desc_tf_idf_document.append(OrderedDict(sorted(terms.items(), key=lambda t: t[1], reverse=True))) 

    for items in desc_tf_idf_document:
        for values in items:
            topN_words.append(values)
            if len(topN_words) == topN:
                break
            # print(values, items[values])

    return topN_words    





cleanwords_document_list = read_document()

documents_name_list = get_documents_name_list(cleanwords_document_list)

tf_document = tf(cleanwords_document_list)

idf_document = idf(tf_document, cleanwords_document_list)

tf_idf_document = tf_idf(idf_document)
topN = 10
topN_keywords = topN_tf_idf(tf_idf_document, documents_name_list, topN) 

print(topN_keywords)


# main()
print("--- %s seconds ---" % (time.time() - start_time))




                
        
