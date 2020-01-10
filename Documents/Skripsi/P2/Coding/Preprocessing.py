# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:26:24 2019

@author: fauzanrahman
"""
import csv
import re
import string
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing:
    
    def main(self, data):
#        print ("Preprocessing")
        
#        print ("\nData Cleansing")
        data_cleansing      = self.dataCleansing(data)
#        print (data_cleansing)
        
#        print ("\nCase Folding")
        data_case_folding   = self.caseFolding(data_cleansing)
#        print (data_case_folding)
        
#        print ("\nTokenizing")
        data_token          = self.tokenizing(data_case_folding)
#        print (data_token)
        
#        print ("\nStopword Removal")
        data_remove_stopword = self.removeStopword(data_token)
#        print (data_remove_stopword)
        
#        print ("\nStemming")
        data_stemming       = self.stemming(data_remove_stopword)
#        print (data_stemming)
    
#        print ("\nTerm")
        term                = self.getTerm(data_stemming)
#        print (term)

        return data_stemming, term
    
    
    def dataCleansing(self, data):
        for i in range(len(data)):
            data[i] = data[i].replace("-", " ")
            data[i] = data[i].replace(",", " ")
            table   = str.maketrans(dict.fromkeys(string.punctuation))
            data[i] = data[i].translate(table)
            data[i] = re.sub(r"(^|\W)\d+", "", data[i])
            
        return data
    
    def caseFolding(self, data):
        return [x.lower() for x in data]
    
    def tokenizing(self, data):
        hasil = []
        for i in data:
            hasil.append(i.split(" "))
        
        return hasil
    
    def removeStopword(self, data):
        stopword = pd.read_csv('Dataset/tala.csv')

        hasil = []
        for i in range(len(data)):
            hasil.append([])
            for j in range(len(data[i])):
                if data[i][j] not in stopword.values and data[i][j] != '':
                    hasil[i].append(data[i][j])
                    
        return hasil
    
    def stemming(self, data):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = stemmer.stem(data[i][j])
        
        return data
    
    def getTerm(self, data):
        hasil = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                hasil.append(data[i][j])
        
        return np.unique(hasil)
    
     
    def exportCSV(self, data, filename):
        with open(filename, 'w+') as csvfile:
            csvWriter = csv.writer(csvfile,delimiter=',')
            csvWriter.writerows(data)
    
    