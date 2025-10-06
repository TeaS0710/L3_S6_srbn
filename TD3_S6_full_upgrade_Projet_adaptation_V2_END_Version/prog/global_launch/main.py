#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import time
import os
import pyttsx3

def coeur_rigolo():
    coeurs = ["bravo", "bravo", "bravo", "vive l'ordinateur", "merci processeur"]
    for i in range(10):
        print(" ".join(coeurs))
        time.sleep(0.2)

def alerte_sonore():
    print("🔥 Félicitations, tout est fini ! 🎉 La tarte est cuite ! 🥧")
    tts = pyttsx3.init()
    tts.say("Félicitations, tout est fini ! La tarte est cuite !")
    tts.runAndWait()

def main():
    # 1) Lancer le 1er script (app1/main.py)
    print("=== Lancement du clustering ===")
    ret1 = subprocess.run(["python3", "main.py"], check=True, cwd="../app1")
    print("=== app1 terminée ===\n")

    # 2) Lancer le 2e script (app2/main.py)
    print("=== Lancement de la création de graphiques ===")
    ret2 = subprocess.run(["python3", "main.py"], check=True, cwd="../app2")
    print("=== app2 terminée ===\n")

    print("=== Tous les traitements sont terminés. ===")
    coeur_rigolo()
    alerte_sonore()

if __name__ == "__main__":
    main()
