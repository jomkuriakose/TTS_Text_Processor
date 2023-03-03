'''
TTS Text Processing Module
Developed at SMV Lab, IIT Madras by Jom Kuriakose, etc...
'''

## Add only the necessary PACKAGES here.

import os
import shutil
import traceback
import pandas as pd
from multiprocessing import Process

import time

# import re
# import json
# import string
# from collections import defaultdict
# import subprocess
# from num_to_words import num_to_word
# from g2p_en import G2p

# set GLOBAL VARIABLES here

## location of phone dictionaries
dictionary_location = "/music/jom/S2S_Project/TTS_Text_Processor/phone_dict"

## list of languages supported by unified parser.
language_list = ['assamese', 'bengali', 'bodo', 'english', 'gujarati', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'rajasthani', 'tamil', 'telugu']
language_list.sort()

class Phone_Dictionary:
    '''
    class to do phone dictionary operations.
    dictionary contains word and its phone representation separated by a tab (\t)
    dictionary stored in self.phone_dictionary

    optional inputs to this module are:
    1) dict_location: dictionary_location
    2) lang_list: language_list
    if not set implicitly the global values will be taken as default

    functions in this module are:
    1) load dictionary
    2) read from dictionary
    3) check for word in dictionary
    4) add new words to dictionary
    5) update dictionary file
    6) delete dictionary entry %% not implemented %% not sure if needed
    ''' 
    def __init__(self, dict_location=None, lang_list=None):
        '''
        select and load dictionaries
        '''
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
        read from dictionary
        '''
        try:
            return self.phone_dictionary[language][word]
        except Exception as e:
            print(traceback.format_exc())
            print(f"word ({word}) not found in language ({language})!!")
            return False
        
    def check_dict_entry(self, language, word):
        '''
        check if the word is present in the dictionary
        '''
        try:
            if word in self.phone_dictionary[language]:
                return True
            else:
                return False
        except Exception as e:
            print(traceback.format_exc())
            return False
    
    def add_to_dict(self, new_dict, language):
        '''
        add the new words to dictionary
        '''
        try:
            if language not in self.phone_dictionary:
                self.phone_dictionary[language] = new_dict
            else:
                self.phone_dictionary[language].update(new_dict)
            # run a non-blocking child process to update the dictionary file
            p = Process(target=self.__update_dict, args=(new_dict, language))
            p.start()
        except Exception as e:
            print(traceback.format_exc())

    def __update_dict(self, dict_to_add, language):
        '''
        update the dictionary file with new words
        '''
        append_string = ""
        for key, value in dict_to_add.items():
            append_string += (str(key) + "\t" + str(value) + "\n")
        
        dict_file = os.path.join(self.dict_location, language)
        
        if os.path.isfile(dict_file):
            # make a copy of the dictionary
            source_dir = os.path.dirname(dict_file)
            dict_file_name = os.path.basename(dict_file)
            temp_file_name = "." + dict_file_name + ".temp"
            temp_dict_file = os.path.join(source_dir, temp_file_name)
            shutil.copy(dict_file, temp_dict_file)
            # append the new words in the dictionary to the temp file
            with open(temp_dict_file, "a") as f:
                f.write(append_string)
            # check if the write is successful and then replace the temp file as the dict file
            try:
                df_orig = pd.read_csv(dict_file, delimiter="\t", header=None, dtype=str)
                df_temp = pd.read_csv(temp_dict_file, delimiter="\t", header=None, dtype=str)
                if len(df_temp) > len(df_orig):
                    os.rename(temp_dict_file, dict_file)
                    print(f"{len(dict_to_add)} new words appended to dictionary: {dict_file}")
            except:
                print(traceback.format_exc())
        else:
            # create a new dictionary
            with open(dict_file, "a") as f:
                f.write(append_string)
            print(f"new dictionary: {dict_file} created with {len(dict_to_add)} words")

## SUDHANSHU
class Word_Parser:
    '''
    class for unified parser, ssn tamil parser and english parser

    inputs to this module are:
    1) tamil_parser_path: ssn_tamil_parser_location
    if not set implicitly the global values will be taken as default
    since unified parser and english parser are python packages, load that in the start of the tts_text_processor module

    functions in this module are:
    1) parse words function with inputs list of words and language
    2) unified parser function - parallelized
    3) english parser function - parallelized
    4) tamil parser function - not parallelized
    '''
    def __init__(self, tamil_parser_path=None):
        '''
        check and load parsers or set parser paths
        '''

## JOHN
class Numerical_Parser:
    '''
    class for parsing number to phone text

    functions in this module are:
    1) parse numbers function with inputs list of numbers and language
    2) number parser - parallelized
    add new modules for different types of numbers like int, float, fraction etc...
    '''
    def __init__(self):
        '''
        check and load defaults if any
        '''

## commands to test the code.

start_time = time.time()
phone_dict = Phone_Dictionary()
end_time = time.time()
print(f"time taken for loading the module: {end_time-start_time:.5f} seconds\n")
start_time = time.time()
print(phone_dict.read_dict_entry("hindi", "अकठोरीकृत"))
end_time = time.time()
print(f"time taken for reading: {end_time-start_time:.5f} seconds\n")
start_time = time.time()
print(phone_dict.read_dict_entry("hindi", "अकठोरीकृतअकठोरीकृत"))
end_time = time.time()
print(f"time taken for reading: {end_time-start_time:.5f} seconds\n")
start_time = time.time()
print(phone_dict.check_dict_entry("hindi", "अकठोरीकृत"))
end_time = time.time()
print(f"time taken for checking: {end_time-start_time:.5f} seconds\n")
start_time = time.time()
print(phone_dict.check_dict_entry("indian", "अकठोरीकृत"))
end_time = time.time()
print(f"time taken for checking: {end_time-start_time:.5f} seconds\n")
start_time = time.time()
print(phone_dict.check_dict_entry("hindi", "अकठोरीकृतअकठोरीकृत"))
end_time = time.time()
print(f"time taken for checking: {end_time-start_time:.5f} seconds")
# new_dict = {}
# new_dict["Jom"] = "Test"
# start_time = time.time()
# print(phone_dict.add_to_dict(new_dict, "hindi"))
# end_time = time.time()
# print(f"time taken for checking: {end_time-start_time:.5f} seconds")
