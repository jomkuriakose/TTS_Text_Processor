    def parsing_parallel(dictn, word):
        dictn[i] = wordparse(i,0,0,1)
    
    def __phonify(self, text, language, gender):
        words = set((" ".join(text)).split(" "))
        print(f"words: {words}")
        non_dict_words = []
        if language in self.phone_dictionary:
            for word in words:
                if word not in self.phone_dictionary[language] and (language == "english" or (not self.__is_english_word(word))):
                    non_dict_words.append(word)
        else:
            non_dict_words = words
        print(f"word not in dict: {non_dict_words}")

        if len(non_dict_words) > 0:
            if(language == 'tamil'):
                os.makedirs("tmp", exist_ok=True)
                timestamp = str(time.time())
                non_dict_words_file = os.path.abspath("tmp/non_dict_words_" + timestamp)
                out_dict_file = os.path.abspath("tmp/out_dict_" + timestamp)
                with open(non_dict_words_file, "w") as f:
                    f.write("\n".join(non_dict_words))
                tamil_parser_cmd = "tamil_parser.sh"
                subprocess.run(["bash", tamil_parser_cmd, non_dict_words_file, out_dict_file, timestamp, "/music/jom/S2S_Project/TTS_Text_Processor/text2phone/Tamil_Parser/ssn_parser"])
                try:
                    df = pd.read_csv(out_dict_file, delimiter="\t", header=None, dtype=str)
                    new_dict = df.dropna().set_index(0).to_dict('dict')[1]
                    print(new_dict)
                except Exception as err:
                    traceback.print_exc()

            elif(language == 'english'):
                new_dict = {}
                for i in range(0,len(non_dict_words)):
                    new_dict[non_dict_words[i]] = self.en_g2p(non_dict_words[i])

            else:
                manager = Manager()
                new_dict = manager.dict()
                job = [Process(target=parsing_parallel, args=(new_dict, i)) for i in non_dict_words]
                _ = [p.start() for p in job]
                _ = [p.join() for p in job]
                # print(Parallel(n_jobs=3)(new_dict_temp[i] = delayed(wordparse)(i,0,0,1) for i in non_dict_words))
                # print(Parallel(n_jobs=3)(delayed(wordparse)(i,0,0,1) for i in non_dict_words))
                # print(Parallel(n_jobs=3)(delayed(wordparse)(i,0,0,1) for i in non_dict_words))
                # for i in non_dict_words:
                #     # print(i)
                #     new_dict_temp[i] = (wordparse(i,0,0,1))

            try:
                # df = pd.read_csv(out_dict_file, delimiter="\t", header=None, dtype=str)
                # new_dict = df.dropna().set_index(0).to_dict('dict')[1]
                print(new_dict)
                if language not in self.phone_dictionary:
                    self.phone_dictionary[language] = new_dict
                else:
                    self.phone_dictionary[language].update(new_dict)
                # run a non-blocking child process to update the dictionary file
                p = Process(target=add_to_dictionary, args=(new_dict, os.path.join(self.dict_location, language)))
                p.start()
            except Exception as err:
                traceback.print_exc()

        # phonify text with dictionary
        text_phonified = []
        for phrase in text:
            phrase_phonified = []
            for word in phrase.split(" "):
                if word in self.phone_dictionary[language]:
                    # if a word could not be parsed, skip it
                    phrase_phonified.append(str(self.phone_dictionary[language][word]))
            # text_phonified.append(self.__post_phonify(" ".join(phrase_phonified),language, gender))
            text_phonified.append(" ".join(phrase_phonified))
        return text_phonified