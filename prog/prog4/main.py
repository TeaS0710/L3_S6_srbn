#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:50:09 2024

@author: ceres
"""
import json
import glob
import csv

def lire_fichier (chemin):
    liste_r=[]
    with open(chemin, newline='') as csv_file: 
        spamreader =csv.reader(csv_file, delimiter=" ", quotechar="\n")
        for row in spamreader:
            liste_r.append(row)
    return liste_r

path_data="../DATA/*/*/*.bio"

for chemin in glob.glob(path_data):
    data=lire_fichier(chemin)
    liste_EN=[]
    for d in data:
        if len(d)>1:
            
            if d[1]!= "O":
                
                liste_EN.append(d[0])

print("Il y a",len(liste_EN),"d’entite«s nomme«es pour le texte",chemin)
  
        
