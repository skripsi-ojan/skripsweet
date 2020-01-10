# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:56:10 2020

@author: fauzanrahman
"""
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

class Pengujian:
    def hitungAkurasi(self, y_test, y_pred):
        hasil = 0
        
        for i in range(len(y_test)):
            if (y_test[i] == y_pred[i]):
                hasil+=1
        
        hasil = hasil/len(y_test) * 100
        return hasil
    
    def confusionMatrix(self, y_test, y_pred):
        hasil = 0
        akurasi = 0
        kelas = 0
        
        actual = y_test
        predict = y_pred
        print("\nConfusion Matrix:")
        hasil = confusion_matrix(actual, predict)
        print(hasil)
        akurasi = accuracy_score(actual, predict)
        print("Akurasi : ", 100 * akurasi, "%")
        kelas = classification_report(actual, predict)
        print("Report : \n", kelas)
        
        return hasil, akurasi, kelas
        
    
    def kFoldCrossValidation(self, kF, dtLatih, klsL):
# =============================================================================
#         np.random.shuffle(dtLatih)
# =============================================================================
        
        kFold       = []
        klsLkFold   = []
        bagi        = len(dtLatih) / kF
        sisa        = len(dtLatih) % kF
        
        adders = 0
        for i in range(kF):
            kFold.append([])
            klsLkFold.append([])
            for j in range(int(bagi)):
                kFold[i].append(dtLatih[adders])
                klsLkFold[i].append(klsL[adders])
                adders +=1
                
        if adders == (len(dtLatih) - sisa):
            for k in range(sisa):
                kFold[i].append(dtLatih[adders+k])
                klsLkFold[i].append(klsL[adders+k])
        
        return np.array(kFold), np.array(klsLkFold)

    
    def bagiDataLatihKFold(self, index, dtL, klsL):
        
        dataLatihBaru   = []
        kelasLatihBaru  = []
        
        for i in range(len(dtL)):
            if (i != index):
                for j in range(len(dtL[i])):
                    dataLatihBaru.append(dtL[i][j])
                
                for j in range(len(klsL[i])):
                    kelasLatihBaru.append(klsL[i][j])
        
        return dataLatihBaru, kelasLatihBaru