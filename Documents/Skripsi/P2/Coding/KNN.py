# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:05:06 2019

@author: fauzanrahman
"""
import numpy as np

class KNN:
    def main(self, k, x_train, y_train, x_test):
        sum_sqrt_latih  = self.getPowerDoc(x_train)
        sum_sqrt_uji    = self.getPowerDoc(x_test)
#        print("\nPerhitungan Vektor")
#        print("\nVektor Data Latih")
#        print(np.power(x_train,2))
#        print("\nVektor Data Uji")
#        print(np.power(x_test,2))
#        print("\nPerkalian Vektor")
#        print(x_train*x_test)
#        print("\nAkar dari Jumlah Vektor Data Latih :", sum_sqrt_latih)
#        print("Akar dari Jumlah Vektor Data Uji :", sum_sqrt_uji)
        x_test  = np.transpose(x_test)
        kelas   = []
        for i in range(len(x_test)):
            _sum                = self.getDocNew(x_train, x_test[i])
#            print("\nJumlah dari Perkalian Vektor :", _sum)
            hasil_cosim         = self.cosim(sum_sqrt_latih, sum_sqrt_uji[i], _sum)
            hasil_cosim_terurut = self.urutkanHasilCosim(hasil_cosim, y_train)
            hasil_cosim_terpilih = self.hapusHasilCosim(hasil_cosim_terurut, k)
#            print("\nCosine Similarity")
#            print(hasil_cosim)
#            print("\nUrutan Cosine Similarity")
#            print(hasil_cosim_terurut)
#            print("\nPemilihan Kelas")
#            print(hasil_cosim_terpilih)
            
            hasil_kelas         = self.voteKelas(hasil_cosim_terpilih)
            kelas.append(hasil_kelas)
            
        return kelas
    
    def voteKelas(self, hasil_cosim):
        kelas_unique = np.unique(hasil_cosim)
        
        ## Inisialisasi Total Kelas dengan 0 {'Negatif': 0, 'Positif': 0}
        total = {}
        for i in kelas_unique:
            total[i] = 0
        
        for i in hasil_cosim:
            total[i] += 1
        
        kelas = max(total, key=total.get)
        
        return kelas
    
    def hapusHasilCosim(self, cosim, k):
        hasil = [cosim[k][1] for k in range(k)]
        return hasil
    
    def urutkanHasilCosim(self, cosim, y_train):
        hasil = []
        
        for i in range(len(cosim)):
            hasil.append([])
            hasil[i].append(cosim[i])
            hasil[i].append(y_train[i])
            
        hasil = np.array(hasil)
        hasil_sort = hasil[hasil[:,0].argsort()[::-1]]
        
        return hasil_sort
    
    def getPowerDoc(self, tfidf):
        tfidf = np.power(tfidf,2)
        tfidf = np.transpose(tfidf)
        
        hasil = [np.sum(i) for i in tfidf]
        hasil = np.sqrt(hasil)

        return hasil
    
    def getDocNew(self, x_train, q):
        doc_new = np.zeros((len(x_train), len(x_train[0])))
        for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                doc_new[i][j] = x_train[i][j] * q[i]
        
        doc_new = np.transpose(doc_new)
        hasil   = [np.sum(i) for i in doc_new]
        
        return hasil
    
    def cosim(self, sum_sqrt_latih, sum_sqrt_uji, _sum):
        hasil = []
        for i in range(len(sum_sqrt_latih)):
            hasil.append(_sum[i] / (sum_sqrt_latih[i] * sum_sqrt_uji))
        
        return hasil