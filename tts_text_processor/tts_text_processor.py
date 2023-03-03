'''
TTS Text Processing Module
Developed at IIT Madras by Jom Kuriakose, ...
'''
import os
import traceback

import re
import json
import string
from collections import defaultdict
import time
import subprocess
import shutil
from multiprocessing import Process
from num_to_words import num_to_word
from g2p_en import G2p
import pandas as pd

# Set global variables here
## location of phone dictionaries
dictionary_location = "/music/jom/S2S_Project/TTS_Text_Processor/phone_dict"
## set the dictionary list to indian and english for using a single dictionary for all Indian languages in unified parser. If set as None, the dictionary list will automatically be read from the dictionary_location
dictionary_list = ["indian", "english"]
## list of languages supported by unified parser. Dictionary list will be subset of this.
language_list = ['assamese', 'bengali', 'bodo', 'english', 'gujarati', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'rajasthani', 'tamil', 'telugu']

class Phone_Dictionary:
    def __init__(self, dict_location=None, dict_list=None, lang_list=None):
        print(f"\nclass Phone_Dictionary :: loading __init__ :: loading phone dictionaries")

        # Unless specified set the default dictionary location
        if dict_location is None:
            dict_location = dictionary_location
        self.dict_location = dict_location
        print(f"dict_location:- {self.dict_location}")

        # Check if the dictionary_list is specified and if not read the list of files and load
        if dict_list is None:
            try:
                dict_list = os.listdir(dict_location)
            except Exception as e:
                print(traceback.format_exc())
                printf("Error:: dictionary loading failed!!")
                return
            dict_list = list(set(dict_list) & set(language_list))
        self.dict_list = dict_list
        print(f"dict_list:- {self.dict_list}")

        # load dictionary from files
        self.phone_dictionary = self.__load_dictionary(self.dict_location, self.dict_list)
    
    def __load_dictionary():

phone_dict=Phone_Dictionary(dict_list=dictionary_list)
# phone_dict=Phone_Dictionary(dict_location='temp')
# phone_dict=Phone_Dictionary()
print("Hello")