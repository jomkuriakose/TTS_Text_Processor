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

# set global variables here
## location of phone dictionaries
dictionary_location = "/music/jom/S2S_Project/TTS_Text_Processor/phone_dict"

## list of languages supported by unified parser.
language_list = ['assamese', 'bengali', 'bodo', 'english', 'gujarati', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'rajasthani', 'tamil', 'telugu']
language_list.sort()

'''
class to do phone dictionary operations
1) load dictionary
2) read from dictionary
3) check for word in dictionary
4) update dictionary entry
5) delete dictionary entry
'''
class Phone_Dictionary:
    '''
    select and load dictionaries
    '''
    def __init__(self, dict_location=None, lang_list=None):
        print(f"\nclass Phone_Dictionary :: loading __init__ :: loading phone dictionaries\n")

        # unless specified set the default dictionary location
        if dict_location is None:
            dict_location = dictionary_location
        self.dict_location = dict_location
        print(f"dict_location:- {self.dict_location}")

        # unless specified set the default language list
        if lang_list is None:
            lang_list = language_list
        lang_list.sort()
        self.lang_list = lang_list
        print(f"lang_list:- {self.lang_list}")

        # read the list of dictionary files and load
        try:
            # select all files in dict_location except hidden files
            dict_list = [file for file in os.listdir(dict_location) if not file.startswith(".")]
        except Exception as e:
            print(traceback.format_exc())
            print("Error:: dictionary loading failed!!")
            return
        self.dict_list = list(set(dict_list) & set(self.lang_list))
        self.dict_list.sort()
        print(f"dict_list:- {self.dict_list}")

        # load dictionary from files
        print(f"\nloading phone dictionaries")
        self.phone_dictionary = {}
        try:
            self.phone_dictionary = self.__load_dictionary()
        except Exception as e:
            print(traceback.format_exc())
            print("Error:: dictionary loading failed!!")
            return
        print(f"phone dictionaries loaded for languages: {[self.phone_dictionary.keys()]}\n")
    
    def __load_dictionary(self):
        '''
        loading the dictionaries from files
        '''
        for language in self.dict_list:
            try:
                dict_file_path = os.path.join(self.dict_location, language)
                df = pd.read_csv(dict_file_path, delimiter="\t", header=None, dtype=str)
                self.phone_dictionary[language] = df.set_index(0).to_dict('dict')[1]
            except Exception as e:
                print(traceback.format_exc())
                print(f"Error:: loading dictionary failed for {language}!!")
                continue
        return self.phone_dictionary
    
    def read_dict_entry(self, language, word):
        '''
        '''
        try:
            return self.phone_dictionary[language][word]
        except Exception as e:
            print(traceback.format_exc())
            print(f"word ({word}) not found in language ({language})!!")
            return False

phone_dict = Phone_Dictionary()
print(phone_dict.read_dict_entry("hindi","अकठोरीकृत"))