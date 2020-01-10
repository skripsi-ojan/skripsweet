# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:55:16 2019

@author: fauzanrahman
"""
import numpy as np

class InformationGain:
    def main(self, kelas, doc_freq, term_latih, term_uji, df, idf):
        pC, logPC                               = self.getPeluangKelas(kelas)
        pBersyarat, logBersyarat                = self.getPeluangBersyarat(kelas, doc_freq, term_latih, False)
        pNegasiBersyarat, logNegasiBersyarat    = self.getPeluangBersyarat(kelas, doc_freq, term_latih, True)
        pTerm                                   = self.getPeluangTerm(df, len(kelas), False)
        pNegasiTerm                             = self.getPeluangTerm(df, len(kelas), True)
        ig1                                     = self.getIG1(pC, logPC)
        ig2                                     = self.getIG23(pBersyarat, logBersyarat, pTerm)
        ig3                                     = self.getIG23(pNegasiBersyarat, logNegasiBersyarat, pNegasiTerm)
            
        ig                                      = self.getIG(ig1, ig2, ig3)
        
        median                                  = np.median(ig)
        ig_terurut                              = self.urutkanIG(ig, term_latih)
        term_latih_new                          = self.hapusIG(ig_terurut, median)
        term_uji_new                            = self.hapusUji(term_latih_new, term_uji)
#        print ("\nPengurutan")
        print (ig_terurut)
        print ("\nThreshold :", median)
        
        return term_latih_new, term_uji_new
    
    def hapusUji(self, term_latih, term_uji):
        term_uji_new = []
        
        for i in range(len(term_uji)):
            for j in range(len(term_latih)):
                if (term_uji[i] == term_latih[j]):
                    term_uji_new.append(term_uji[i])
        
        return term_uji_new
    
    def hapusIG(self, ig_terurut, median):
        term_latih = []
        for i in range(len(ig_terurut)):
            if (float(ig_terurut[i][0]) < median):
                term_latih.append(ig_terurut[i][1])
                
        return term_latih
        
    def urutkanIG(self, ig, term):
        hasil = []
        for i in range(len(ig)):
            hasil.append([])
            hasil[i].append(ig[i])
            hasil[i].append(term[i])
        
        hasil = np.array(hasil)
        hasil_sort = hasil[hasil[:,0].argsort()[::-1]]
        
        return hasil_sort
                
    def getIG(self, ig1, ig2, ig3):
        hasil = np.zeros(len(ig2))
        for i in range(len(hasil)):
            hasil[i] = ig1 + ig2[i] + ig3[i]
        
        return hasil 
    
    def getIG23(self, pBersyarat, logBersyarat, pTerm):
        
        hasil = np.zeros(len(pTerm))
        for i in range(len(pTerm)):
            for kls in pBersyarat:
                hasil[i] += pBersyarat[kls][i] * logBersyarat[kls][i]
            hasil[i] *= pTerm[i]
        
        return hasil
        
    def getIG1(self, pC, logPC):
        hasil = 0
        for kls in pC:
            hasil += pC[kls] * logPC[kls]
        
        hasil *= -1
        
        return hasil
        
    def getPeluangTerm(self, df, panjangDoc, negasi):
        pTerm = np.zeros(len(df))
        if (negasi == False):
            for i in range(len(df)):
                pTerm[i] = df[i] / np.sum(df)
        else:
            for i in range(len(df)):
                pTerm[i] = (panjangDoc - df[i]) / np.sum(df)
        
        return pTerm
        
    def getPeluangKelas(self, kelas):
        kls_unique = np.unique(kelas)
        
        ## Inisialisasi total kelas {positif : 0, negatif : 0}
        pC = {}
        for i in kls_unique:
            pC[i] = 0
        
        ## Menjumlahkan kelas
        for i in kelas:
            pC[i] +=1
        
        ## Mencari peluang kelas dan log
        logPC = pC.copy()
        for i in pC:
            pC[i] /= len(kelas)
            logPC[i] = np.log2(pC[i])
        
        return pC, logPC
    
    def getPeluangBersyarat(self, kelas, doc_freq, term, negasi):
        kls_unique = np.unique(kelas)
        
        ## Inisialisasi total kelas {positif : 0, negatif : 0}
        total = {}
        for i in kls_unique:
            total[i] = 0
        
        ## Menjumlahkan kelas
        for i in kelas:
            total[i] +=1
        
        ## Inisialisasi peluang bersyarat dengan list {'Positif': [], 'Negatif': []}
        pBersyarat = {}
        for i in kelas:
            pBersyarat[i] = []
        
        ## Inisialisasi log peluang bersyarat dengan list {'Positif': [], 'Negatif': []}
        logBersyarat = {}
        for i in kelas:
            logBersyarat[i] = []
        
        for i in range(len(term)):
            ## Inisialisasi jumlah kelas {positif : 0, negatif : 0}
            jumlah = {}
            for kls in kls_unique:
                jumlah[kls] = 0
            
            ## Mencari jumlah kelas 
            for j in range(len(kelas)):
                jumlah[kelas[j]] += int(doc_freq[i][j])
            
            ## Menghitung peluang bersyarat dan log peluang bersyarat masing masing kelas
            for kls in kls_unique:
                if (negasi == False):
                    peluang     = jumlah[kls] / total[kls]
                else :
                    peluang     = (total[kls]-jumlah[kls]) / total[kls]
                
                if (peluang == 0):
                    log_peluang = 0
                else :
                    log_peluang = np.log2(peluang)
                
                pBersyarat[kls].append(peluang)
                logBersyarat[kls].append(log_peluang)
                
        return pBersyarat, logBersyarat