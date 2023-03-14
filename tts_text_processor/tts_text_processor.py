'''
TTS Text Processing Module
Developed at SMV Lab, IIT Madras -- Prof. Hema A. Murthy

Contributors
1) Jom Kuriakose
2) Sudhanshu Srivastava
## Add your names here.
'''

## Add only the necessary PACKAGES here.
import re
import json
import string
from collections import defaultdict
import time
import subprocess
import shutil
from multiprocessing import Process
import traceback
from num_to_words import num_to_word
from g2p_en import G2p
import pandas as pd
from indic_unified_parser.uparser import wordparse
from joblib import Parallel, delayed
import multiprocessing
from multiprocess import Process, Manager
num_cores = multiprocessing.cpu_count()
import os
import shutil
import traceback
import pandas as pd
import multiprocessing as mp
from g2p_en import G2p
from multiprocessing import Process
from indic_unified_parser.uparser import wordparse
from itertools import repeat, chain
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

## update dictionary file
update_dict_flag = True

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
    6) edit a entry in dictionary
    7) delete dictionary entry
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
    
    def add_to_dict(self, language, new_dict):
        '''
        add the new words to dictionary
        '''
        try:
            if language not in self.phone_dictionary:
                self.phone_dictionary[language] = new_dict
            else:
                self.phone_dictionary[language].update(new_dict)
            if update_dict_flag:
                # run a non-blocking child process to update the dictionary file
                p = Process(target=self.__update_dict_file, args=(new_dict, language, 'add'))
                p.start()
        except Exception as e:
            print(traceback.format_exc())
    
    def edit_dict(self, language, edit_dict):
        '''
        edit already existing dictionary entries
        '''
        try:
            if language not in self.phone_dictionary:
                self.phone_dictionary[language] = edit_dict
            else:
                self.phone_dictionary[language].update(edit_dict)
            if update_dict_flag:
                # run a non-blocking child process to update the dictionary file
                p = Process(target=self.__update_dict_file, args=(edit_dict, language, 'replace'))
                p.start()
        except Exception as e:
            print(traceback.format_exc())

    def delete_dict_entry(self, language, del_dict):
        '''
        delete the entries in del_dict from the phone dictionary
        '''
        for key in del_dict.keys():
            try:
                del self.phone_dictionary[language][key]
            except Exception as e:
                print(traceback.format_exc())
                continue
        if update_dict_flag:
            try:
                # run a non-blocking child process to update the dictionary file
                p = Process(target=self.__update_dict_file, args=(del_dict, language, 'delete'))
                p.start()
            except Exception as e:
                print(traceback.format_exc())

    def __update_dict_file(self, update_dict, language, operation):
        '''
        update the dictionary file. operations: add, delete, replace
        '''
        dict_file = os.path.join(self.dict_location, language)
        if operation == "add":
            '''
            add the new words to the dictionary file without checking if the entry already exists
            '''
            append_string = ""
            for key, value in update_dict.items():
                append_string += (str(key) + "\t" + str(value) + "\n")
            
            # update the dict file
            if os.path.isfile(dict_file):
                # make a copy of the dictionary
                source_dir = os.path.dirname(dict_file)
                dict_file_name = os.path.basename(dict_file)
                temp_file_name = "." + dict_file_name + ".temp"
                temp_dict_file = os.path.join(source_dir, temp_file_name)
                print(f"copy dict_file: {dict_file} to temp_dict_file: {temp_dict_file}")
                shutil.copyfile(dict_file, temp_dict_file)
                # append the new words in the dictionary to the temp file
                with open(temp_dict_file, "a") as f:
                    f.write(append_string)
                # check if the write is successful and then replace the temp file as the dict file
                try:
                    df_orig = pd.read_csv(dict_file, delimiter="\t", header=None, dtype=str)
                    df_temp = pd.read_csv(temp_dict_file, delimiter="\t", header=None, dtype=str)
                    if len(df_temp) > len(df_orig):
                        os.rename(temp_dict_file, dict_file)
                        print(f"{len(update_dict)} new words appended to dictionary: {dict_file}")
                except:
                    print(traceback.format_exc())
            else:
                # create a new dictionary
                with open(dict_file, "a") as f:
                    f.write(append_string)
                print(f"new dictionary: {dict_file} created with {len(update_dict)} words")
        elif operation == "replace":
            '''
            update the already existing dictionary entries along with adding new ones to the dictionary
            '''
            update_string = ""
            for key, value in self.phone_dictionary[language].items():
                update_string += (str(key) + "\t" + str(value) + "\n")
            
            # update the dict file
            if os.path.isfile(dict_file):
                # make a copy of the dictionary
                source_dir = os.path.dirname(dict_file)
                dict_file_name = os.path.basename(dict_file)
                temp_file_name = "." + dict_file_name + ".temp"
                temp_dict_file = os.path.join(source_dir, temp_file_name)
                print(f"copy dict_file: {dict_file} to temp_dict_file: {temp_dict_file}")
                shutil.copyfile(dict_file, temp_dict_file)
                # append the new words in the dictionary to the temp file
                with open(temp_dict_file, "w") as f:
                    f.write(update_string)
                # check if the write is successful and then replace the temp file as the dict file
                try:
                    df_orig = pd.read_csv(dict_file, delimiter="\t", header=None, dtype=str)
                    df_temp = pd.read_csv(temp_dict_file, delimiter="\t", header=None, dtype=str)
                    if len(df_temp) >= len(df_orig):
                        os.rename(temp_dict_file, dict_file)
                        print(f"updated the dictionary: {dict_file}")
                except:
                    print(traceback.format_exc())
            else:
                # create a new dictionary
                with open(dict_file, "a") as f:
                    f.write(update_string)
                print(f"new dictionary: {dict_file} created with {len(update_dict)} words")
        elif operation == "delete":
            '''
            delete entries from the dictionary if present
            '''
            update_string = ""
            for key, value in self.phone_dictionary[language].items():
                update_string += (str(key) + "\t" + str(value) + "\n")
            
            # update the dict file
            if os.path.isfile(dict_file):
                # make a copy of the dictionary
                source_dir = os.path.dirname(dict_file)
                dict_file_name = os.path.basename(dict_file)
                temp_file_name = "." + dict_file_name + ".temp"
                temp_dict_file = os.path.join(source_dir, temp_file_name)
                print(f"copy dict_file: {dict_file} to temp_dict_file: {temp_dict_file}")
                shutil.copyfile(dict_file, temp_dict_file)
                # append the new words in the dictionary to the temp file
                with open(temp_dict_file, "w") as f:
                    f.write(update_string)
                # check if the write is successful and then replace the temp file as the dict file
                try:
                    df_orig = pd.read_csv(dict_file, delimiter="\t", header=None, dtype=str)
                    df_temp = pd.read_csv(temp_dict_file, delimiter="\t", header=None, dtype=str)
                    if len(df_temp) <= len(df_orig):
                        os.rename(temp_dict_file, dict_file)
                        print(f"updated the dictionary: {dict_file}")
                except:
                    print(traceback.format_exc())
            else:
                # dictionary doesn't exist
                print(f"dictionary: {dict_file} doesn't exist\ndelete from dictionary failed\nadd to dictionary to create the dictionary")
        else:
            print(f"operation ({operation}) not supported\npossible operations are (add, replace, delete)\ndictionary file not updated")

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
    5) mapping from cmu to cls function
    6) mapping from cls to char text function
    '''
    def __init__(self, dict_location=None, tamil_parser_path=None):
        '''
        check and load parsers or set parser paths
        '''
        # dictionary to store non_dictionary_words
        self.new_dict = {}
        # if dict_location is None:
        #     dict_location = "/music/jom/S2S_Project/TTS_Text_Processor/phone_dict"
        # self.dict_location = dict_location
        # self.phone_dictionary = {}
        # load dictionary for all the available languages
        # for dict_file in os.listdir(dict_location):
        #     try:
        #         if dict_file.startswith("."):
        #             # ignore hidden files
        #             continue
        #         language = dict_file
        #         dict_file_path = os.path.join(dict_location, dict_file)
        #         df = pd.read_csv(dict_file_path, delimiter=" ", header=None, dtype=str)
        #         self.phone_dictionary[language] = df.set_index(0).to_dict('dict')[1]
        #     except Exception as e:
        #         print(traceback.format_exc())

        # print("Phone dictionary loaded for the following languages:", list(self.phone_dictionary.keys()))

        self.g2p = G2p()
        print('Loading G2P model... Done!')
        # Mapping between the cmu phones and the iitm cls
        self.cmu_2_cls_map = {
            "AA" : "aa",
            "AA0" : "aa",
            "AA1" : "aa",
            "AA2" : "aa",
            "AE" : "axx",
            "AE0" : "axx",
            "AE1" : "axx",
            "AE2" : "axx",
            "AH" : "a",
            "AH0" : "a",
            "AH1" : "a",
            "AH2" : "a",
            "AO" : "ax",
            "AO0" : "ax",
            "AO1" : "ax",
            "AO2" : "ax",
            "AW" : "ou",
            "AW0" : "ou",
            "AW1" : "ou",
            "AW2" : "ou",
            "AX" : "a",
            "AY" : "ei",
            "AY0" : "ei",
            "AY1" : "ei",
            "AY2" : "ei",
            "B" : "b",
            "CH" : "c",
            "D" : "dx",
            "DH" : "d",
            "EH" : "ee",
            "EH0" : "ee",
            "EH1" : "ee",
            "EH2" : "ee",
            "ER" : "a r",
            "ER0" : "a r",
            "ER1" : "a r",
            "ER2" : "a r",
            "EY" : "ee",
            "EY0" : "ee",
            "EY1" : "ee",
            "EY2" : "ee",
            "F" : "f",
            "G" : "g",
            "HH" : "h",
            "IH" : "i",
            "IH0" : "i",
            "IH1" : "i",
            "IH2" : "i",
            "IY" : "ii",
            "IY0" : "ii",
            "IY1" : "ii",
            "IY2" : "ii",
            "JH" : "j",
            "K" : "k",
            "L" : "l",
            "M" : "m",
            "N" : "n",
            "NG" : "ng",
            "OW" : "o",
            "OW0" : "o",
            "OW1" : "o",
            "OW2" : "o",
            "OY" : "ei",
            "OY0" : "ei",
            "OY1" : "ei",
            "OY2" : "ei",
            "P" : "p",
            "R" : "r",
            "S" : "s",
            "SH" : "sh",
            "T" : "tx",
            "TH" : "t",
            "UH" : "u",
            "UH0" : "u",
            "UH1" : "u",
            "UH2" : "u",
            "UW" : "uu",
            "UW0" : "uu",
            "UW1" : "uu",
            "UW2" : "uu",
            "V" : "w",
            "W" : "w",
            "Y" : "y",
            "Z" : "z",
            "ZH" : "sh",
        }

        # Mapping between the iitm cls and iitm char
        self.cls_2_chr_map = {
            "aa" : "A",
            "ii" : "I",
            "uu" : "U",
            "ee" : "E",
            "oo" : "O",
            "nn" : "N",
            "ae" : "ऍ",
            "ag" : "ऽ",
            "au" : "औ",
            "axx" : "अ",
            "ax" : "ऑ",
            "bh" : "B",
            "ch" : "C",
            "dh" : "ध",
            "dx" : "ड",
            "dxh" : "ढ",
            "dxhq" : "T",
            "dxq" : "D",
            "ei" : "ऐ",
            "ai" : "ऐ",
            "eu" : "உ",
            "gh" : "घ",
            "gq" : "G",
            "hq" : "H",
            "jh" : "J",
            "kh" : "ख",
            "khq" : "K",
            "kq" : "क",
            "ln" : "ൾ",
            "lw" : "ൽ",
            "lx" : "ള",
            "mq" : "M",
            "nd" : "न",
            "ng" : "ङ",
            "nj" : "ञ",
            "nk" : "Y",
            "nw" : "ൺ",
            "nx" : "ण",
            "ou" : "औ",
            "ph" : "P",
            "rq" : "R",
            "rqw" : "ॠ",
            "rw" : "ർ",
            "rx" : "र",
            "sh" : "श",
            "sx" : "ष",
            "th" : "थ",
            "tx" : "ट",
            "txh" : "ठ",
            "wv" : "W",
            "zh" : "Z",
        }

        # Multilingual support for OOV characters
        oov_map_json_file = 'multilingualcharmap.json'
        with open(oov_map_json_file, 'r') as oov_file:
            self.oov_map = json.load(oov_file)

    # def cls2chr(self, text):
    #     while(len(cmustring)):


    def en_g2p(self, word, op_format='normal', sep=False):
        phn_out = self.g2p(word)
        # import pdb;pdb.set_trace()
        # return in cmu format if required
        if op_format == 'cmu' or op_format == 'CMU':
            if sep==False:
                return ("".join(phn_out)).strip().replace(" ", "")
            else:
                return (" ".join(phn_out)).strip()
            
        if op_format=='cls' or op_format=='CLS':
            for i, phn in enumerate(phn_out):
                if phn in self.cmu_2_cls_map.keys():
                    phn_out[i] = self.cmu_2_cls_map[phn]
                else:
                    pass
        # print(f"phn_out: {phn_out}")
        # iterate over the string list and replace each word with the corresponding value from the dictionary
        
        ## Converting to char from cls
        else:
            for i, phn in enumerate(phn_out):
                if phn in self.cmu_2_cls_map.keys():
                    phn_out[i] = self.cmu_2_cls_map[phn]
                    # cls_out = self.cmu_2_cls_map[phn]
                    if phn_out[i] in self.cls_2_chr_map.keys():
                        phn_out[i] = self.cls_2_chr_map[phn_out[i]]
                    else:
                        pass
                else:
                    pass  # ignore words that are not in the dictionary
            # print(f"i: {i}, phn: {phn}, cls_out: {cls_out}, phn_out: {phn_out[i]}")
        
        ## By default, it will return in character format
        if sep==False:
            return ("".join(phn_out)).strip().replace(" ", "")
        else:
            return (" ".join(phn_out)).strip()
    
    def english_parse(self, non_dict_words,op_format='normal', sep=False):
        for i in range(0,len(non_dict_words)):
            self.new_dict[non_dict_words[i]] = self.en_g2p(non_dict_words[i], op_format, sep)

    def tamil_parse(self, non_dict_words):
        os.makedirs("tmp", exist_ok=True)
        timestamp = str(time.time())
        non_dict_words_file = os.path.abspath("tmp/non_dict_words_" + timestamp)
        out_dict_file = os.path.abspath("tmp/out_dict_" + timestamp)
        with open(non_dict_words_file, "w") as f:
            f.write("\n".join(non_dict_words))
        tamil_parser_cmd = "../text2phone/Tamil_Parser/tamil_parser.sh"
        subprocess.run(["bash", tamil_parser_cmd, non_dict_words_file, out_dict_file, timestamp, "/tts/srivastava/text_normalization/TTS_Text_Processor/text2phone/Tamil_Parser/ssn_parser"])
        try:
            df = pd.read_csv(out_dict_file, delimiter="\t", header=None, dtype=str)
            self.new_dict = df.dropna().set_index(0).to_dictprint(chkwrdp.phonify('This is a sample sentence', 'english', 'cls', True))('dict')[1]
            print(self.new_dict)
        except Exception as err:
            traceback.print_exc()
    
    def cls2char(self, phones, sep=False):
        for i in range(0, len(phones)):
            if phones[i] in self.cls_2_chr_map.keys():
                phones[i] = self.cls_2_chr_map[phones[i]]
        return phones
    
    def regular_parse(self, non_dict_words, op_format='normal', sep=False):
        # print(Parallel(n_jobs=3)(delayed(wordparse)(i,0,0,1) for i in non_dict_words))
        # def ppp(word):
        #     self.new_dict[word] = wordparse(word,0,0,1)   
        # Parallel(n_jobs=3)(delayed(self.ppp)(i) for i in non_dict_words)
        worker_pool = mp.Pool(7)
        parser_outputs = worker_pool.starmap(wordparse, zip(non_dict_words, repeat(0), repeat(0), repeat(1)))
        
        if op_format == 'char' or op_format == 'Char' or op_format == 'CHAR':
            for i in range(0, len(non_dict_words)):
                temp_lst = self.cls2char(parser_outputs[i].split(), sep)
                if sep == False:
                   self.new_dict[non_dict_words[i]] = ("".join(temp_lst)).strip()
                    # self.new_dict[non_dict_words[i]] = self.cls2char(parser_outputs[i].split(), sep).replace(" ", "")
                else:
                    self.new_dict[non_dict_words[i]] = (" ".join(temp_lst)).strip()
                    # self.new_dict[non_dict_words[i]] = self.cls2char(parser_outputs[i], sep)
                # self.new_dict[non_dict_words[i]] = cls2char(parser_outputs[i], sep)
        else:
            for i in range(0, len(non_dict_words)):
                if sep == False:
                    self.new_dict[non_dict_words[i]] = parser_outputs[i].replace(" ", "")
                else:
                    self.new_dict[non_dict_words[i]] = parser_outputs[i]

    def phonify(self, text, language, op_format='normal', sep=False):
        words = text.split()
        # print("check1")
        # exit(0)
        # words = set((" ".join(text)).split(" "))
        # print(f"words: {words}")
        # non_dict_words = []
        # if language in self.phone_dictionary:
        #     for word in words:
        #         if word not in self.phone_dictionary[language] and (language == "english" or (not self.__is_english_word(word))):
        #             non_dict_words.append(word)
        # else:
        #     non_dict_words = words
        non_dict_words = words
        if len(non_dict_words) > 0:
            if language=='tamil' or language=='Tamil':
                self.tamil_parse(non_dict_words)
            elif language=='english' or language=='English':
                self.english_parse(non_dict_words, op_format, sep)
            else:
                self.regular_parse(non_dict_words, op_format, sep)
            # try:
                # df = pd.read_csv(out_dict_file, delimiter="\t", header=None, dtype=str)
                # new_dict = df.dropna().set_index(0).to_dict('dict')[1]
                # print("till here")
                # print(self.new_dict)
                # if language not in self.phone_dictionary:
                #     self.phone_dictionary[language] = self.new_dict
                # else:
                #     self.phone_dictionary[language].update(self.new_dict)
                # run a non-blocking child process to update the dictionary file
                # p = Process(target=add_to_dictionary, args=(self.new_dict, os.path.join(self.dict_location, language)))
                # p.start()
            # except Exception as err:
            #     traceback.print_exc()
        # text_phonified = []
        # for phrase in text:
        #     phrase_phonified = []
        #     for word in phrase.split(" "):
        #         if word in self.phone_dictionary[language]:
        #             # if a word could not be parsed, skip it
        #             phrase_phonified.append(str(self.phone_dictionary[language][word]))
        #     # text_phonified.append(self.__post_phonify(" ".join(phrase_phonified),language, gender))
        #     text_phonified.append(" ".join(phrase_phonified))
        return self.new_dict

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

## all - Need major discussion here.
class Input_Str_Processor:
    '''
    class for processing input string to find the type of words, like numbers, equations, date, abbreviations, math symbols, other symbols, brackets, 200c-200e, etc...

    functions in this module are:
    1) parse string to find types of strings
    add new modules for detecting different types of string.

    new classes for processing these has to be developed based on what all we can detect.
    '''
    def __init__(self):
        '''
        check and load defaults if any
        '''

## commands to test the code.
# phone_dict = Phone_Dictionary()

# chkwrdp = Word_Parser()
# print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi'))
# print(chkwrdp.phonify('This is a sample sentence', 'english'))
# print(chkwrdp.phonify('நீங்கள் ஆங்கிலம் பேசுகிறீர்களா', 'tamil'))

# start_time = time.time()
# phone_dict = Phone_Dictionary()
# end_time = time.time()
# print(f"time taken for loading the module: {`end_time-start_time:.5f} seconds\n")
# start_time = time.time()
# print(phone_dict.read_dict_entry("hindi", "अकठोरीकृत"))
# end_time = time.time()
# print(f"time taken for reading: {end_time-start_time:.5f} seconds\n")
# start_time = time.time()
# print(phone_dict.read_dict_entry("hindi", "अकठोरीकृतअकठोरीकृत"))
# end_time = time.time()
# print(f"time taken for reading: {end_time-start_time:.5f} seconds\n")
# start_time = time.time()
# print(phone_dict.check_dict_entry("hindi", "अकठोरीकृत"))
# end_time = time.time()
# print(f"time taken for checking: {end_time-start_time:.5f} seconds\n")
# start_time = time.time()
# print(phone_dict.check_dict_entry("indian", "अकठोरीकृत"))
# end_time = time.time()
# print(f"time taken for checking: {end_time-start_time:.5f} seconds\n")
# start_time = time.time()
# print(phone_dict.check_dict_entry("hindi", "अकठोरीकृतअकठोरीकृत"))
# end_time = time.time()
# print(f"time taken for checking: {end_time-start_time:.5f} seconds")

# new_dict = {}
# new_dict["Jom"] = "Test"
# start_time = time.time()
# print(phone_dict.add_to_dict("hindi", new_dict))
# end_time = time.time()
# print(f"time taken for checking: {end_time-start_time:.5f} seconds")

## Directions to use the Parsing functions
## For english you can return the output in CLS/CHAR/CMU format with/without separator
## For Hindi and other languages only cls and char are available with/without separator
## For Tamil, only char is available
## English sentence - Default is char
## Hindi - Defalut is CLS
## True - space would be the separotor, otherwise no separator

if __name__=="__main__":
    start_time = time.time()
    chkwrdp = Word_Parser()
    end_time = time.time()
    print(f"time taken for loading: {end_time-start_time:.5f} seconds")
    # print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi'))
    # print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi',True))
    # print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi','char'))
    print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi','cmu',True))
    start_time = time.time()
    # print(chkwrdp.phonify('This is a sample sentence','english', sep=True))
    # print(chkwrdp.phonify('ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.', 'hindi','char',True))
    end_time = time.time()
    print(f"time taken for loading: {end_time-start_time:.5f} seconds")
    # print(chkwrdp.phonify('This is a sample sentence', 'english', 'char', True))
    print(chkwrdp.phonify('This is a sample sentence', 'english', 'cmu', True))
    # print(chkwrdp.phonify('நீங்கள் ஆங்கிலம் பேசுகிறீர்களா', 'tamil'))