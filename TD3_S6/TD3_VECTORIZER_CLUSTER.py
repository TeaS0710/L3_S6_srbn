#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""


import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric 
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
from collections import OrderedDict


def lire_fichier (chemin):
    with open(chemin) as json_data: 
        texte =json.load(json_data)
    return texte

def nomfichier(chemin):
    nomfich= chemin.split("/")[-1]
    nomfich= nomfich.split(".")
    nomfich= ("_").join([nomfich[0],nomfich[1]])
    return nomfich
    
 

chemin_entree =



for subcorpus in glob.glob(path_copora):
#    print("SUBCORPUS***",subcorpus)
    liste_nom_fichier =[]
    for path in glob.glob("%s/AIMARD-TRAPPEURS_MOD/AIMARD_les-trappeurs_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json"%subcorpus):
#        print("PATH*****",path)
        
        nom_fichier = nomfichier(path)
#        print(nom_fichier)
        liste=lire_fichier(path)
        
        
#### FREQUENCE ########
        
        dic_mots={}
        i=0
    
        
        for mot in liste: 
            
            if mot not in dic_mots:
                dic_mots[mot] = 1
            else:
                dic_mots[mot] += 1
        
        i += 1

        new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))
        
        freq=len(dic_mots.keys())
        

        Set_00 = set(liste)
        Liste_00 = list(Set_00)
        dic_output = {}
        liste_words=[]
        matrice=[]
        
        for l in Liste_00:
                
            if len(l)!=1:
                liste_words.append(l)
        

        try:
            words = np.asarray(liste_words)
            for w in words:
                liste_vecteur=[]
            
                    
                for w2 in words:
                
                        V = CountVectorizer(ngram_range=(2,3), analyzer='char')
                        X = V.fit_transform([w,w2]).toarray()
                        distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)            
                        liste_vecteur.append(distance_tab1[0][1])
                    
                matrice.append(liste_vecteur)
            matrice_def=-1*np.array(matrice)
           
                  
            affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) 
    
            affprop.fit(matrice_def)
            for cluster_id in np.unique(affprop.labels_):
                exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
                cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
                cluster_str = ", ".join(cluster)
                cluster_list = cluster_str.split(", ")
                            
                Id = "ID "+str(i)
                for cle, dic in new_d.items(): 
                    if cle == exemplar:
                        dic_output[Id] ={}
                        dic_output[Id]["Centroïde"] = exemplar
                        dic_output[Id]["Freq. centroide"] = dic
                        dic_output[Id]["Termes"] = cluster_list
                
                i=i+1
            #    print(dic_output)
            

        except :        
            print("**********Non OK***********", path)

    
            liste_nom_fichier.append(path)
            
            
            continue 


    
