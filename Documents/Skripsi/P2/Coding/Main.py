# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:25:42 2019

@author: fauzanrahman
"""
import pandas as pd
from Preprocessing import Preprocessing
from TermWeighting import TermWeighting
from InformationGain import InformationGain
from KNN import KNN
from Pengujian import Pengujian

class MainProgram():
    FILENAME_LATIH  = 'Datasett/Data Latihh.xlsx'
    FILENAME_UJI    = 'Datasett/Data Ujii.xlsx'
    
    def main(self):
        data_latih  = pd.read_excel(self.FILENAME_LATIH)
        data_uji    = pd.read_excel(self.FILENAME_UJI)
        
        pre         = Preprocessing()
        data_latih_pre, term_latih  = pre.main(data_latih['Text'].values)
        data_uji_pre, term_uji      = pre.main(data_uji['Text'].values)
    
        tw          = TermWeighting()
        dataset     = tw.gabungData(data_latih_pre, data_uji_pre)
#        print ("\nTF dan DF")
#        print ("\nRAW TF")
        raw_tf      = tw.getRawTF(dataset, term_latih)
#        print (raw_tf)
#        print ("\nDF")
        doc_freq    = tw.getDocumentFrecuency(raw_tf)
#        print (doc_freq)
        df, idf     = tw.getIdf(doc_freq[:, :len(data_latih)])
#        print ("\nDF Data Latih Total : ", sum(df))
        
        ig          = InformationGain()
#        print ("\nHasil Information Gain")
        term_latih_new, term_uji_new = ig.main(data_latih['Class'], doc_freq, term_latih, term_uji, df, idf)
    
#        print ("\nData Latih")
        print (term_latih_new)
#        print ("\nData Uji")
        print (term_uji_new)
        
#        print ("\nTerm Weighting")
#        print ("\nRAW TF")
        raw_tf_new          = tw.getRawTF(dataset, term_latih_new)
#        print (raw_tf_new)
#        print ("\nDF")
        doc_freq_new        = tw.getDocumentFrecuency(raw_tf_new)
#        print(doc_freq_new)
#        print ("\nLOG TF")
        log_normalization   = tw.logNormalization(raw_tf_new)
#        print (log_normalization)
#        print ("\nIDF")
        df_new, idf_new     = tw.getIdf(doc_freq_new[:, :len(data_latih)])
#        print (idf_new)
#        print ("\nTF IDF")
        tf_idf              = tw.getTfIdf(log_normalization, idf_new)
#        print (tf_idf)
        
        knn             = KNN()
        x_train         = tf_idf[:, :len(data_latih)]
        y_train         = data_latih['Class']
        x_test          = tf_idf[:, len(data_latih):len(data_latih)+len(data_uji)]
        k               = 29
        
        kelas           = knn.main(k, x_train, y_train, x_test)
        print("\nK :", k)
        print("\nKelas :", kelas)
        
        
        pj              = Pengujian()
#        akurasi         = pj.hitungAkurasi(data_uji['Class'], kelas)
        conf            = pj.confusionMatrix(data_uji['Class'], kelas)
#        print("Akurasi :", akurasi, " %")
        
if __name__ == "__main__":
    mn = MainProgram()
    mn.main()