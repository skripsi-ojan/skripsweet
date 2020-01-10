# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:34:11 2019

@author: fauzanrahman
"""
import numpy as np

class TermWeighting:
    
    def getTfIdf(self, log_normalization, idf):
        tfidf = np.zeros((len(log_normalization), len(log_normalization[0])))
        
        for i in range(len(tfidf)):
            for j in range(len(tfidf[i])):
                tfidf[i][j] = log_normalization[i][j] * idf[i]
        
        return tfidf
    
        
    def logNormalization(self, doc):
        normalization = doc.copy()
        
        for i in range(len(normalization)):
            for j in range(len(normalization[i])):
                if (normalization[i][j] == 0):
                    normalization[i][j] = 0
                else:
                    normalization[i][j] = np.log10(normalization[i][j]) + 1
        
        return normalization
    
    def gabungData(self, data_latih, data_uji):
        dataset = data_latih.copy()
        for i in data_uji:
            dataset.append(i)
        
        return dataset
    
    def getRawTF(self, data, term):
        hasil = np.zeros((len(term), len(data)))
        
        for i in range(len(term)):
            for j in range(len(data)):
                for k in range(len(data[j])):
                    if term[i] == data[j][k]:
                        hasil[i][j] +=1

        return hasil
    
    def getDocumentFrecuency(self, doc):
        hasil = doc.copy()
        for i in range(len(hasil)):
            for j in range(len(hasil[i])):
                if hasil[i][j] >= 1:
                    hasil[i][j] = 1
                    
        return hasil
    
    def getIdf(self, docs):
        idf  = np.zeros(len(docs))
        df   = np.zeros(len(docs))
        
        for i in range(len(docs)):
            df[i] = np.sum(docs[i])
            
            if (df[i] != 0):
                idf[i]      = np.log10(len(docs[i])/df[i])  
            else:
                idf[i]      = 0
                
        return df, idf