# TTS Text processing package

## TODO
1. Sudhanshu - Integrating python unified parser (https://pypi.org/project/indic-unified-parser/) in parallel in old_text_preprocessor.py
2. John - Write a function to convert numbers to cls using ipa to cls map. Write it as a class and have the map in the init function.

## Steps
1. 

## How to use
```python
from old_text_preprocess_code import TTSPreprocessor, TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor_VTT, TTSDurAlignPreprocessor_VTT, CharTextPreprocessor_VTT

tts_preprocessor = TTSPreprocessor()
durali_preprocessor = TTSDurAlignPreprocessor()
char_preprocessor = CharTextPreprocessor()
tts_preprocessor_vtt = TTSPreprocessor_VTT()
durali_preprocessor_vtt = TTSDurAlignPreprocessor_VTT()
char_preprocessor_vtt = CharTextPreprocessor_VTT()

tts_preprocessor.preprocess("ये नया सिस्टम ऐसा कॉन्टेट लिख सकता है जो बहुत ही सटीक होता है और इंसानों के लिखे जैसा ही प्रतीत होता है.", "hindi", "male")
```