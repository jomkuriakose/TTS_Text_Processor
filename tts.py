'''
TTS API
Developed by Arun Kumar A(CS20S013) - November 2022
This is a python Flask API. Reverse proxy is set up in apache conf file
'''
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
# import pdb
# from pprint import pprint
import math
import sys
import os
os.environ['PYTHONPATH']="/speech/arun/tts/hifigan"
from flask import Flask, jsonify, request, render_template
from espnet2.bin.tts_inference import Text2Speech
import time
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from itertools import repeat, chain
import numpy as np
import os
import json
from env import AttrDict
import torch
import traceback
import base64
from collections import defaultdict
from pathlib import Path
import datetime
import logging
import subprocess
import requests
from io import StringIO
from webvtt import WebVTT, Caption, read_buffer
from collections import defaultdict

from synthesise_phrase import PhraseSynthesizer
# from synthesise_segments import SegmentSynthesizer
from synthesise_segments_wo_preproc import SegmentSynthesizer_VTT
from text_preprocess_w_eng_preproc_out import TTSPreprocessor, TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor_VTT, TTSDurAlignPreprocessor_VTT, CharTextPreprocessor_VTT
from api_processor import APIProcessor

MODELS_DIR="/speech/released_models"
DEVICE="cuda"
if DEVICE == "cuda":
    MAX_WORKERS=8
    N_GPUS=4
    START_GPU=3
else:
    MAX_WORKERS=24

SAMPLING_RATE=22050
TTS_DEFAULT_MODEL = defaultdict(lambda: "fastspeech2_hs")
TTS_DEFAULT_MODEL.update({
    "english/male" : "fastspeech2_tf",
    "english/female" : "fastspeech2_tf",
    "urdu/male" : "fastspeech2_tf_char",
    "urdu/female" : "fastspeech2_tf_char",
    "punjabi/male" : "fastspeech2_tf_char",
    "punjabi/female" : "fastspeech2_tf_char",
})
VOCODER_DEFAULT_MODEL = defaultdict(lambda: "hifigan")
VOCODER_DEFAULT_MODEL.update({
    "bodo/female" : "aryan/hifigan",
    "gujarati/male" : "aryan/hifigan",
    "gujarati/female" : "aryan/hifigan",
    "manipuri/male" : "aryan/hifigan",
    "manipuri/female" : "aryan/hifigan",
    "odia/male" : "aryan/hifigan",
    "odia/female" : "aryan/hifigan",
    "punjabi/male" : "aryan/hifigan",
    "punjabi/female" : "aryan/hifigan",
    "rajasthani/male" : "aryan/hifigan",
    "rajasthani/female" : "aryan/hifigan",
    "english/male" : "aryan/hifigan",
    "english/female" : "aryan/hifigan",
})

# DRAVIDIAN_LANGUAGES = set(["tamil"])
DRAVIDIAN_LANGUAGES = set(["tamil", "malayalam", "kannada", "telugu"])
# DRAVIDIAN_LANGUAGES = set(["tamil", "telugu"])
INDO_ARYAN_LANGUAGES = set(["hindi", "gujarati", "marathi", "odia", "bengali", "punjabi", "urdu", "rajasthani", "assamese", "bodo", "manipuri", "english"])
# INDO_ARYAN_LANGUAGES = set(["hindi", "english"])
# INDO_ARYAN_LANGUAGES = set(["hindi", "gujarati", "marathi", "bengali", "punjabi", "urdu", "english"])
LANG_FAMILIES = set(["aryan", "dravidian"])

logger = logging.getLogger("TTS_API")
logger.setLevel(logging.INFO)

app = Flask(__name__)
requests_log = {}
whitelist = []
waitlist = []
blacklist = []

# Add to whitelist
whitelist.append('10.21.238.100') # Jom Lab System
whitelist.append('10.24.6.85') # D1
whitelist.append('10.24.6.151') # asr/tts webserver ip
whitelist.append('121.242.232.133') # Tenet group - Greeshma

tts_model_map = defaultdict(lambda: defaultdict(lambda: {}))
vocoder_model_map = defaultdict(lambda: defaultdict(lambda: {}))
tts_model_description = defaultdict(lambda: {})
vocoder_model_description = defaultdict(lambda: {})
worker_pool = None
api_processor = APIProcessor()
tts_preprocessor = TTSPreprocessor()
durali_preprocessor = TTSDurAlignPreprocessor()
char_preprocessor = CharTextPreprocessor()
tts_preprocessor_vtt = TTSPreprocessor_VTT()
durali_preprocessor_vtt = TTSDurAlignPreprocessor_VTT()
char_preprocessor_vtt = CharTextPreprocessor_VTT()


def get_formatted_time(time_in_seconds):
    m, s = divmod(time_in_seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_hifigan_vocoder(model_file, config_file, device):
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    device=torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(model_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_fastspeech2_model(model_file, config_file, device):
    return Text2Speech(train_config=config_file, model_file=model_file, device=device)

def load_tacotron2_model():
    # TODO
    return None

def load_tts_model(model_file, config_file, device):
    return load_fastspeech2_model(model_file, config_file, device)

def load_vocoder_model(model_file, config_file, device):
    return load_hifigan_vocoder(model_file, config_file, device)

def load_models_from_directory(models_dir, model_type):
    model_description = defaultdict(lambda: defaultdict(lambda: {}))
    model_map = None
    model_file_name = None
    config_file_name = None
    if model_type == "tts":
        model_loading_function = load_tts_model
        model_map = tts_model_map
        model_file_name = "model.pth"
        config_file_name = "config.yaml"
    elif model_type == "vocoder":
        model_loading_function = load_vocoder_model
        model_map = vocoder_model_map
        model_file_name = "generator"
        config_file_name = "config.json"
    else:
        logger.error(f"Unsupported model type: {model_type}. Supported types are tts and vocoder. Models won't be loaded.")
        return None
    model_i = 0
    for gender in os.listdir(models_dir):
        gender_dir = os.path.join(models_dir, gender)
        for language in os.listdir(gender_dir):
            if language in set(list(LANG_FAMILIES) + list(DRAVIDIAN_LANGUAGES) + list(INDO_ARYAN_LANGUAGES)):
                language_dir = os.path.join(gender_dir, language)
                if os.path.isdir(language_dir):
                    for model_name in os.listdir(language_dir):
                        model_dir_path = os.path.join(language_dir, model_name)
                        if model_name.startswith(".") or not os.path.isdir(model_dir_path):
                            continue
                        model_file = Path(os.path.join(model_dir_path, model_file_name))
                        config_file = Path(os.path.join(model_dir_path, config_file_name))
                        if not config_file.exists():
                            logger.warning(f"{config_file_name} is missing in {model_dir_path}. Ignoring this model.")
                            continue
                        if not model_file.exists():
                            logger.warning(f"{model_file_name} is missing in {model_dir_path}. Ignoring this model.")
                            continue

                        device = DEVICE
                        if device == "cuda":
                            device = device + ":" + str(START_GPU+(model_i%N_GPUS))
                        
                        logger.info(f"Loading {model_type} with config {str(config_file.resolve())} and model file {str(model_file.resolve())}")
                        model = model_loading_function(str(model_file.resolve()), str(config_file.resolve()), device)
                        model.device = device
                        model_map[language][gender][model_name] = model
                        logger.info(f"{gender}/{language}/{model_name} loaded in device {device}")
                        model_i += 1

                        # load description of the model
                        if not bool(model_description[language][gender]):
                            caption = None
                            try:                            
                                with open(os.path.join(language_dir, "caption"), "r") as f:
                                    caption = f.read().strip()
                            except Exception as e:
                                logger.error(e)
                            model_description[language]["caption"] = caption if caption is not None else language
                        if "models" not in model_description[language][gender]:
                            model_description[language][gender]["models"] = {}
                        description = ""
                        try:                            
                            with open(os.path.join(model_dir_path, "description"), "r") as f:
                                description = f.read().strip()
                        except:
                            pass
                        model_description[language][gender]["models"][model_name] = {"description": description}
    return model_map, model_description

def synthesise_speech_from_vtt(json_data):
    with app.app_context():
        tmp_dir = None
        vtt_content = json_data["input"]
        # logger.info(f"vtt_content: {vtt_content}")
        language = json_data["lang"].lower()
        logger.info(f"language: {language}")
        gender = json_data["gender"].lower()
        logger.info(f"gender: {gender}")
        alpha = float(json_data["alpha"] if "alpha" in json_data else 1)
        decode_conf = {"alpha": alpha}
        logger.info(f"decode_conf: {decode_conf}")
        tts_model_variant = json_data["tts_model"] if "tts_model" in json_data else None
        vocoder_model_variant = json_data["vocoder_model"] if "vocoder_model" in json_data else None
        key = f"{language}/{gender}"
        if tts_model_variant is None:
            tts_model_variant = TTS_DEFAULT_MODEL[key]            
        if vocoder_model_variant is None:
            vocoder_model_variant = VOCODER_DEFAULT_MODEL[key]
            if "/" in vocoder_model_variant:
                vocoder_lang = vocoder_model_variant.split("/")[0]
                vocoder_model_variant = vocoder_model_variant.split("/")[1]
            else:
                vocoder_lang = language

        if tts_model_variant not in tts_model_map[language][gender]:
            return jsonify(status='failure', reason=f"Unsupported TTS model type {tts_model_variant} for the given gender {gender}"), tmp_dir
        if vocoder_model_variant not in vocoder_model_map[vocoder_lang][gender]:
            return jsonify(status='failure', reason=f"Unsupported vocoder model type {vocoder_model_variant} for the given gender {gender}."), tmp_dir
        
        logger.info(f"tts_model_variant: {tts_model_variant}")
        logger.info(f"vocoder_model_variant: {vocoder_model_variant}")

        text2speech = tts_model_map[language][gender][tts_model_variant]
        generator = vocoder_model_map[vocoder_lang][gender][vocoder_model_variant]

        #create vtt object
        vtt = read_buffer(StringIO(vtt_content))
        segments = [caption.text for caption in vtt.captions]
        # logger.info(f"segments: {segments}")
        
        splits = min(len(segments), MAX_WORKERS)
        if (tts_model_variant == 'fastspeech2_tf'):
            preprocessor = tts_preprocessor_vtt
        elif (tts_model_variant == 'fastspeech2_hs'):
            preprocessor = durali_preprocessor_vtt
        elif (tts_model_variant == 'fastspeech2_tf_char'):
            preprocessor = char_preprocessor_vtt
        else:
            logger.info(f"TTS model variant: {tts_model_variant}; not available!!")
        # pdb.set_trace()
        # logger.info(f"Before preprocessor:\n segments: {segments}, language: {language}, gender: {gender}, preprocessor: {preprocessor}")
        preprocessed_text, phrases = preprocessor.preprocess(segments, language, gender)
        # logger.info(f"After preprocessor:\n preprocessed_text: {preprocessed_text}, phrases: {phrases}")
        # outputs = worker_pool.starmap(SegmentSynthesizer_VTT.synthesise_segments,
        #                             zip(np.array_split(np.array(preprocessed_text), splits), repeat(text2speech), repeat(generator), repeat(MAX_WAV_VALUE), repeat(decode_conf)))
        splits = min(len(preprocessed_text), MAX_WORKERS)
        outputs = worker_pool.starmap(SegmentSynthesizer_VTT.synthesise_segments,
                                    zip(np.array_split(np.array(preprocessed_text), splits), repeat(text2speech), repeat(generator), repeat(MAX_WAV_VALUE), repeat(decode_conf)))
        audio_segments = list(chain(*outputs))
        timestamp=str(datetime.datetime.now())
        for spl_char in ['-', ' ', ':', '.']:
            timestamp = timestamp.replace(spl_char, '_')
        
        tmp_dir=f"tmp/data/data_{timestamp}"
        # create tmp directory
        os.makedirs(tmp_dir, exist_ok = True)

        #generate segment wise wav files
        audio_segments_wav_bytes = []
        for i, audio_segment in enumerate(audio_segments):
            segment_out_file = os.path.join(tmp_dir, f"segment_{i}.wav")
            write(segment_out_file, SAMPLING_RATE, audio_segment)
            segment_wav_bytes = base64.b64encode(open(segment_out_file, "rb").read())
            audio_segments_wav_bytes.append(segment_wav_bytes.decode('utf-8'))
        
        # generate full audio if needed
        if "fullaudio" in json_data and json_data["fullaudio"]:
            fullAudio = np.concatenate(audio_segments)
            output_file = os.path.join(tmp_dir, "gen.wav")
            write(output_file, SAMPLING_RATE, fullAudio)
            full_audio_wav_bytes = base64.b64encode(open(output_file, "rb").read())
            return jsonify(status="success", audio=full_audio_wav_bytes.decode('utf-8'), segments=audio_segments_wav_bytes), tmp_dir

        return jsonify(status="success", segments=audio_segments_wav_bytes), tmp_dir

def synthesise_speech_from_text(json_data):
    with app.app_context():
        tmp_dir = None
        text = json_data["input"]
        if not isinstance(text,str):
            input_type = type(text)
            return jsonify(status='failure', reason=f"Unsupported input type {input_type}. Input text should be in string format."), tmp_dir
        language = json_data["lang"].lower()
        gender = json_data["gender"].lower()
        alpha = float(json_data["alpha"] if "alpha" in json_data else 1)
        decode_conf = {"alpha": alpha}
        tts_model_variant = json_data["tts_model"] if "tts_model" in json_data else None
        vocoder_model_variant = json_data["vocoder_model"] if "vocoder_model" in json_data else None
        key = f"{language}/{gender}"
        if tts_model_variant is None:
            tts_model_variant = TTS_DEFAULT_MODEL[key]            
        if vocoder_model_variant is None:
            vocoder_model_variant = VOCODER_DEFAULT_MODEL[key]
            if "/" in vocoder_model_variant:
                vocoder_lang = vocoder_model_variant.split("/")[0]
                vocoder_model_variant = vocoder_model_variant.split("/")[1]
            else:
                vocoder_lang = language

        if tts_model_variant not in tts_model_map[language][gender]:
            return jsonify(status='failure', reason=f"Unsupported TTS model type {tts_model_variant} for the given gender {gender}"), tmp_dir
        if vocoder_model_variant not in vocoder_model_map[vocoder_lang][gender]:
            return jsonify(status='failure', reason=f"Unsupported vocoder model type {vocoder_model_variant} for the given gender {gender}."), tmp_dir
        

        text2speech = tts_model_map[language][gender][tts_model_variant]
        generator = vocoder_model_map[vocoder_lang][gender][vocoder_model_variant]
        if (tts_model_variant == 'fastspeech2_tf'):
            preprocessor = tts_preprocessor
        elif (tts_model_variant == 'fastspeech2_hs'):
            preprocessor = durali_preprocessor
        elif (tts_model_variant == 'fastspeech2_tf_char'):
            preprocessor = char_preprocessor
        else:
            logger.info(f"TTS model variant: {tts_model_variant}; not available!!")
        preprocessed_text, phrases = preprocessor.preprocess(text, language, gender)
        print(f"preprocessed_text: {preprocessed_text}, phrases: {phrases}")
        splits = min(len(preprocessed_text), MAX_WORKERS)
        outputs = worker_pool.starmap(PhraseSynthesizer.synthesise_phrase,
                                    zip(np.array_split(np.array(preprocessed_text), splits), repeat(text2speech), repeat(generator), repeat(MAX_WAV_VALUE), repeat(decode_conf)))
        audio_segments = list(chain(*outputs))
        timestamp=str(datetime.datetime.now())
        for spl_char in ['-', ' ', ':', '.']:
            timestamp = timestamp.replace(spl_char, '_')
        
        tmp_dir=f"tmp/data/data_{timestamp}"
        # create tmp directory
        os.makedirs(tmp_dir, exist_ok = True)

        fullAudio = np.concatenate(audio_segments)
        output_file = os.path.join(tmp_dir, "gen.wav")
        write(output_file, SAMPLING_RATE, fullAudio)
        full_audio_wav_bytes = base64.b64encode(open(output_file, "rb").read())

        vtt = WebVTT()
        last_end_time = 0
        for audio_segment, phrase in zip(audio_segments, phrases):
            start_time = last_end_time
            duration = len(audio_segment) / SAMPLING_RATE
            end_time = start_time + duration
            last_end_time = end_time
            start_milli = int((start_time%1)*1000)
            end_milli = int((end_time%1)*1000)
            caption = Caption(
                    get_formatted_time(int(start_time)) + "." + f"{start_milli:03d}",
                    get_formatted_time(int(end_time)) + "." + f"{end_milli:03d}",
                    phrase
            )
            vtt.captions.append(caption)
        
        if "segmentwise" in json_data and json_data["segmentwise"].lower() == "true":
            #generate segment wise wav files
            audio_segments_wav_bytes = []
            for i, audio_segment in enumerate(audio_segments):
                segment_out_file = os.path.join(tmp_dir, f"segment_{i}.wav")
                write(segment_out_file, SAMPLING_RATE, audio_segment)
                segment_wav_bytes = base64.b64encode(open(segment_out_file, "rb").read())
                audio_segments_wav_bytes.append(segment_wav_bytes.decode('utf-8'))
            return jsonify(status="success", audio=full_audio_wav_bytes.decode('utf-8'), segments=audio_segments_wav_bytes, vtt=vtt.content), tmp_dir

        return jsonify(status="success", audio=full_audio_wav_bytes.decode('utf-8'), vtt=vtt.content), tmp_dir

def api_call(json_data):
    with app.app_context():
        # print(json_data)
        input_json = api_processor.preprocess(json_data)
        # print(input_json)
        out, tmp_dir = synthesise_speech_from_text(input_json)
        # print(out)
        resp = out.get_json()
        # print(resp)
        # print(resp.keys())
        output = api_processor.postprocess(resp, json_data)
        # print(output)
        return output

def make_dummy_calls():
    # dummy calls as a hack to initalize cuda processes so that requests are served fast
    logger.info("Making dummy calls...")
    try:
        # have a dummy text for all languages and use it
        with open("sample_text", "r") as f:
            sample_text = {}
            for line in f.readlines():
                line = line.strip()
                lang, text = line.split(" ", 1)
                sample_text[lang] = text
            for lang in sample_text:
                if lang in tts_model_description:
                    for gender in ("male", "female"):
                        if gender in tts_model_description[lang]:
                            logger.info(f"Making dummy call for {lang}/{gender}")
                            json_data = {
                                "input": sample_text[lang],
                                "lang": lang,
                                "gender": gender
                            }
                            synthesise_speech_from_text(json_data)
            logger.info("All dummy calls has been made")
    except:
        traceback.print_exc()

def setup_app():
    global tts_model_map, vocoder_model_map, tts_model_description, vocoder_model_description, worker_pool
    tts_model_map, tts_model_description = load_models_from_directory(os.path.join(MODELS_DIR, "tts"), "tts")
    vocoder_model_map, vocoder_model_description = load_models_from_directory(os.path.join(MODELS_DIR, "vocoder"), "vocoder")
    
    tts_supported_languages = set(tts_model_description.keys())
    if "dravidian" in tts_supported_languages:
        tts_supported_languages.update(DRAVIDIAN_LANGUAGES)
    if "aryan" in tts_supported_languages:
        tts_supported_languages.update(INDO_ARYAN_LANGUAGES)
    vocoder_supported_languages = set(vocoder_model_description.keys())
    if "dravidian" in vocoder_supported_languages:
        vocoder_supported_languages.update(DRAVIDIAN_LANGUAGES)
    if "aryan" in vocoder_supported_languages:
        vocoder_supported_languages.update(INDO_ARYAN_LANGUAGES)
    supported_languages = tts_supported_languages.intersection(vocoder_supported_languages)
    logger.info(f"TTS supported languages: {tts_supported_languages}")
    logger.info(f"Vocoder supported languages: {vocoder_supported_languages}")

    # remove unnecessary models
    for language in tts_supported_languages.difference(supported_languages).difference(set(["aryan", "dravidian"])):
        tts_model_map.pop(language, None)
        tts_model_description.pop(language, None)

    for language in vocoder_supported_languages.difference(supported_languages).difference(set(["aryan", "dravidian"])):
        vocoder_model_map.pop(language, None)
        vocoder_model_description.pop(language, None)

    for language in supported_languages:
        if "male" in tts_model_map[language]:
            remove = (language in INDO_ARYAN_LANGUAGES and "aryan" in vocoder_model_map and "male" not in vocoder_model_map["aryan"])
            remove = remove or (language in DRAVIDIAN_LANGUAGES and "male" not in vocoder_model_map["dravidian"])
            remove = remove and ("male" not in vocoder_model_map[language])
            if remove:
                tts_model_map[language].pop("male", None)
                tts_model_description[language].pop("male", None)
        if "male" in vocoder_model_map[language]:
            remove = (language in INDO_ARYAN_LANGUAGES and "male" not in tts_model_map["aryan"])
            remove = remove or (language in DRAVIDIAN_LANGUAGES and "male" not in tts_model_map["dravidian"])
            remove = remove and ("male" not in tts_model_map[language])
            if remove:
                vocoder_model_map[language].pop("male", None)
                vocoder_model_description[language].pop("male", None)
        if "female" in tts_model_map[language]:
            remove = (language in INDO_ARYAN_LANGUAGES and "female" not in vocoder_model_map["aryan"])
            remove = remove or (language in DRAVIDIAN_LANGUAGES and "female" not in vocoder_model_map["dravidian"])
            remove = remove and ("female" not in vocoder_model_map[language])
            if remove:
                tts_model_map[language].pop("female", None)
                tts_model_description[language].pop("female", None)
        if "female" in vocoder_model_map[language]:
            remove = (language in INDO_ARYAN_LANGUAGES and "female" not in tts_model_map["aryan"])
            remove = remove or (language in DRAVIDIAN_LANGUAGES and "female" not in tts_model_map["dravidian"])
            remove = remove and ("female" not in tts_model_map[language])
            if remove:
                vocoder_model_map[language].pop("female", None)
                vocoder_model_description[language].pop("female", None)

    # model_description = tts_model_description
    logger.info(f"Models loaded for the following languages: {str(supported_languages)}")

    mp = torch.multiprocessing.get_context("spawn")
    worker_pool = mp.Pool(MAX_WORKERS)
    
    if DEVICE=="cuda":
        make_dummy_calls()
    logger.info("Server initialized")

# setup the application
setup_app()

@app.route('/', strict_slashes=False)
def home():
    return render_template("index.html")

@app.route('/langs', methods=['GET', 'POST'], strict_slashes=False)
def list_supported_languages():
    return jsonify(tts=tts_model_description, vocoder=vocoder_model_description)

@app.route('/vtt-to-speech', methods=['GET', 'POST'], strict_slashes=False)
def vtt_to_speech():
    logger.info("--------------------------------")
    
    client_ip = request.environ['HTTP_X_FORWARDED_FOR']
    current_time = datetime.datetime.now()
    logger.info(f"ip: {client_ip} at {current_time}")

    tmp_dir=None
    try:
        json_data = request.get_json()
        logger.info(f"request received")
        out, tmp_dir = synthesise_speech_from_vtt(json_data)
        return out
    except Exception as err:
        logger.error(traceback.format_exc())
        return jsonify(status="failure", reason=str(err))
    finally:
        # release resources and remove temp files
        # clear temp files in a background process
        if tmp_dir is not None:
            subprocess.Popen(["rm","-rf",tmp_dir])

@app.route('/tts', methods=['GET', 'POST'], strict_slashes=False)
def tts():
    # Request filtering part
    logger.info("--------------------------------")
    
    client_ip = request.environ['HTTP_X_FORWARDED_FOR']
    current_time = datetime.datetime.now()
    logger.info(f"ip: {client_ip} at {current_time}")

    # Reads the request and adds to the request_log

    # requests_log[client_ip]['count'] -- counts the number of requests from client_ip
    # requests_log[client_ip]['times'] -- keeps the time of requests in a list
    # requests_per_minute -- actually not a rate of requests per minute but more like the number of requests in the past 1 minute

    if client_ip in requests_log:
        if (current_time - requests_log[client_ip]['times'][0]).days == 1: # resets the number of requests counter each day
            requests_log[client_ip]['count'] = 1
            requests_log[client_ip]['times'] = [current_time]
            requests_per_minute = 1
            logger.info(f"**NewDay**\nclient_ip: {client_ip}, requests_log[client_ip]['count']: {requests_log[client_ip]['count']}, requests_in_a_minute: {requests_per_minute}")
        else:
            requests_log[client_ip]['count'] += 1
            requests_log[client_ip]['times'].append(current_time)
            requests_per_minute = sum(1 for t in requests_log[client_ip]['times'] if (current_time - t).total_seconds() < 60)
            logger.info(f"**NewReq**\nclient_ip: {client_ip}, requests_log[client_ip]['count']: {requests_log[client_ip]['count']}, requests_in_a_minute: {requests_per_minute}")
    else:
        requests_log[client_ip] = {'count': 1, 'times': [current_time]}
        requests_per_minute = 1
        logger.info(f"**New IP**\nclient_ip: {client_ip}, requests_log[client_ip]['count']: {requests_log[client_ip]['count']}, requests_in_a_minute: {requests_per_minute}")
    
    # Check if the ip is black listed

    if client_ip in blacklist:
        logger.info(f"**Blocked**\n IP: {client_ip}, Time: {current_time}")
        return jsonify(status="failure", reason="Too many requests. Blacklisted. Please mail smtiitm@gmail.com for renewing the access.")
    
    # Check only the requests from ips not in whitelist to see if they are within the limits. Per day number of requests limit is set as 500 and per minute number of requests limit is set as 30.

    if client_ip not in whitelist:
        # check for per day limit breach
        if requests_log[client_ip]['count'] > 500:
            logger.info(f"**Over Day Limit**\nRequest Blocked. IP: {client_ip} at {current_time}")
            if client_ip in waitlist:
                blacklist.append(client_ip)
                logger.info(f"**BlackListed**\nIP: {client_ip} added to black list at {current_time}")
            else:
                waitlist.append(client_ip)
                logger.info(f"**WaitListed**\nIP: {client_ip} added to wait list at {current_time}")
            return jsonify(status="failure", reason="Too many requests per day. Please try again later.")
        else:
            # check for per minute limit breach
            if requests_per_minute > 30:
                logger.info(f"**Over Minute Limit**\nRequest Blocked. IP: {client_ip} at {current_time}")
                waitlist.append(client_ip)
                logger.info(f"**WaitListed**\nIP: {client_ip} added to wait list at {current_time}")
                return jsonify(status="failure", reason="Too many requests per minute. Please try again later.")
            else:
                logger.info(f"**Request Served**\nIP: {client_ip} at {current_time}")
    
    # TTS part

    tmp_dir=None
    try:
        json_data = request.get_json()
        logger.info(f"request received")
        logger.debug(f"{json_data}")
        out, tmp_dir = synthesise_speech_from_text(json_data)
        return out
    except Exception as err:
        logger.error(traceback.format_exc())
        return jsonify(status="failure", reason=str(err))
    finally:
        # release resources and remove temp files
        # clear temp files in a background process
        if tmp_dir is not None:
            subprocess.Popen(["rm","-rf",tmp_dir])

@app.route('/api', methods=['GET', 'POST'], strict_slashes=False)
def api():
    logger.info("--------------------------------")
    
    client_ip = request.environ['HTTP_X_FORWARDED_FOR']
    current_time = datetime.datetime.now()
    logger.info(f"ip: {client_ip} at {current_time}")

    tmp_dir=None
    try:
        json_data = request.get_json()
        logger.info(f"request received")
        logger.debug(f"{json_data}")
        out = api_call(json_data)
        return out
    except Exception as err:
        logger.error(traceback.format_exc())
        return jsonify(status="failure", reason=str(err))
    finally:
        # release resources and remove temp files
        # clear temp files in a background process
        if tmp_dir is not None:
            subprocess.Popen(["rm","-rf",tmp_dir])

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.    
    app.run(host='0.0.0.0', port=6793)
