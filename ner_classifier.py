# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:47:27 2019

@author: Bertha
"""
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import spacy
import en_core_web_sm
import pandas as pd
import numpy as np
import sys
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
from copy import deepcopy
from sklearn.externals import joblib

class Classifier:
    
    def __init__(self):
        self.train_data = pd.read_csv("training_data_ner.csv", encoding = "utf-8", header = 0)
        self.embeddings_dict = {}
        with open("glove.6B.100d.txt", 'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        
    def featureGenerate(self, question, label = None):
    
        nlp = en_core_web_sm.load()
        doc = nlp(u'' + question)
        sent = list(doc.sents)[0]
        
        noWH = True
    
        q_word = ""
        q_pos = ""
        root = ""
        root_pos = ""
        root_lemma = ""
        wh_word = ""
        wh_head = ""
        wh_bigram = []
        wh_pos = ""
        wh_head_pos = ""
        subj = ""
        subj_pos = ""
        subj_ner = ""
    
        for token in sent:
        
            #binary question
            if token.pos_ == "AUX" and token.i == 0:
                q_word = token.lemma_
                q_pos = token.pos_
            
            #wh-question
            if token.tag_ in ("WDT","WP","WP$","WRB") and noWH:
                wh_word = token.lemma_
                wh_pos = token.tag_
                next_token = doc[token.i + 1]
                wh_head = next_token.lemma_
                wh_head_pos = next_token.tag_
                wh_bigram = [wh_word, wh_head]
                noWH = False
            
            if token.dep_ == "ROOT":
                root = token.text
                root_pos = token.tag_
                root_lemma = token.lemma_
        
            if token.dep_ == "nsubj":
                if subj != "":
                    subj += " "
                subj += token.text
                subj_pos = token.tag_
                subj_ner = token.ent_type_
        
        return {
        "question": question,
        "q_word": q_word,
        "q_pos": q_pos,
        "root": root,
        "root_pos": root_pos,
        "root_lemma": root_lemma,
        "wh_word": wh_word,
        "wh_head": wh_head,
        "wh_bigram": wh_bigram,
        "wh_pos": wh_pos,
        "wh_head_pos": wh_head_pos,
        "subj": subj,
        "subj_pos": subj_pos,
        "subj_ner": subj_ner,
        "label": label,
        }
        
    def transform(self, X_train, X_predict):

        X_cols = list(set(list(X_train.columns) + list(X_predict.columns)))

        data_train = {}
        for col in X_cols:
            if col not in X_train:
                data_train[col] = [0 for i in range(len(X_train.index))]
            else:
                data_train[col] = list(X_train[col])

        X_train = pd.DataFrame(data_train)
        X_train = csr_matrix(X_train)
    
        data_predict = {}
        for col in X_cols:
            if col not in X_predict:
                data_predict[col] = [0 for i in range(len(X_predict.index))]
            else:
                data_predict[col] = list(X_predict[col])

        X_predict = pd.DataFrame(data_predict)
        X_predict = csr_matrix(X_predict)

        return X_train, X_predict
        
    def getDummies(self, data):
        
        embedding_df = pd.DataFrame()
        for index, row in data.iterrows():
            try:
                if row.get("wh_word") == "what" and row.get("wh_head") == "be":
                    word = row.get("subj")
                    embedding_df = embedding_df.append(pd.Series(self.embeddings_dict[word.encode('utf-8')]), ignore_index=True)
                else:
                    word = row.get("wh_head")
                    embedding_df = embedding_df.append(pd.Series(self.embeddings_dict[word.encode('utf-8')]), ignore_index=True)
            except:
                embedding_df = embedding_df.append(pd.Series(np.array([0.0]*100)), ignore_index=True)
        data.pop("wh_head")
        data.pop("subj")
        df = pd.get_dummies(data)
        return pd.concat([df,embedding_df], axis=1)
    
    def filtering(self, data):
    
        true_labels = data.pop("label")
        questions = data.pop("question")
        data.pop("root")
        data.pop("root_lemma")
        data.pop("q_word")
        data.pop("q_pos")
        #data.pop("wh_head")
        data.pop("wh_bigram")
        #data.pop("subj")
        data.pop("subj_ner")
    
        return questions, true_labels, data
        
    def prepocess_question(self, question):
        
        questions = [question]
        cols = ["question",
            "q_word",
            "q_pos",
            "root",
            "root_pos",
            "root_lemma",
            "wh_word",
            "wh_head",
            "wh_bigram",
            "wh_pos",
            "wh_head_pos",
            "subj",
            "subj_pos",
            "subj_ner",
            "label"]
    
        test_data = pd.DataFrame(columns = cols)
        for question in questions:
            row = self.featureGenerate(question)
            test_data = test_data.append(row, ignore_index = True)
        
        return test_data
    
    def WrittenRules(self, data):
    
        for index, row in data.iterrows():
            try:                    
                if row.get("wh_word") == "who":
                    if data.loc[index, "predict"] not in ("PERSON","ORG","NORG"):
                        data.loc[index, "predict"] = "PERSON"
                        
                if row.get("wh_word") == "where":
                    if data.loc[index, "predict"] not in ("LOC","GPE","FAC"):
                        data.loc[index, "predict"] = "LOC"
                    
                if row.get("wh_word") == "how":
                    if row.get("wh_head") in ("many"):
                        data.loc[index, "predict"] = "QUANTITY"
                    elif row.get("wh_head") == "much" and row.get("root_lemma") in ("pay","cost","be"):
                        data.loc[index, "predict"] = "MONEY"
                    elif row.get("wh_head") == "old":
                        data.loc[index, 'predict'] = "DATE"

                if row.get("wh_word") == "what" and row.get("wh_head") == "be":
                    if row.get("subj") in ("currency", "money", "salary"):
                        data.loc[index, "predict"] = "MONEY"
                    if row.get("subj") in ("city","state","country","continent","ocean"):
                        data.loc[index, "predict"] = "LOC"
                        
                if row.get("q_pos") == "AUX":
                    data.loc[index, "wh_word"] = "BINARY"
            except:
                pass
    
        return data
 
    @ignore_warnings(category=ConvergenceWarning)
    def SVM(self,X_train, true_labels, X_predict):
        
        lin_clf = LinearSVC()
        lin_clf.fit(X_train, true_labels)
        predict_labels = lin_clf.predict(X_predict)
        return predict_labels
    
    def predict(self, question):
        
        test_data = self.prepocess_question(question)
        train_data = self.train_data
        train_questions, train_true_labels, filtered_train_data = self.filtering(deepcopy(train_data))
        test_questions, test_true_labels, filtered_test_data = self.filtering(deepcopy(test_data))
    
        X_train = self.getDummies(filtered_train_data)
        X_test = self.getDummies(filtered_test_data)
    
        X_train, X_test = self.transform(X_train, X_test)
        predict_labels = self.SVM(X_train, train_true_labels, X_test)
        test_data["predict"] = predict_labels
        test_data = self.WrittenRules(test_data)
        
        return (test_data["wh_word"][0], test_data["predict"][0])
    
if __name__ == "__main__":
    
    #train = pd.read_csv("training_ner.csv", header = 0)
    #labels = []
    #questions = []
    #for index, row in train.iterrows():
        #labels.append(row.get("label"))
        #questions.append(row.get("question"))
    with open("test1.txt", encoding = "utf-8") as file:
        questions = [line.strip() for line in file]
    acc = 0
    model = Classifier()    
    for i in range(len(questions)):
        print(questions[i])
        label = model.predict(questions[i])
        print(label)
        #if label[1] == labels[i]:
            #acc += 1
    print(acc/len(questions))
    

        
        