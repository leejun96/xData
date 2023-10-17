# create a hosted microservice to deploy an Automatic Speech Recognition
# (ASR) AI model that can be used to transcribe any audio files.
# upload files: https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
from jiwer import wer
import pickle
import soundfile as sf
import io
import librosa
import json
from io import BytesIO
import uvicorn
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uuid

app = FastAPI()

''' UploadFile uses spooled memory which is ideal for large files
Attributes of UploadFile :
    filename: original name of uploaded filename [type: str]
    content-type: content i.e. MIME/ media type [type: str]
    file: actual Python file that can be passed directly to other functions or libraries [type: SpooledTemporaryFile] 
'''
# https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile
app = FastAPI()


@app.post("/asr/")
async def create_upload_files(files: list[UploadFile]):
    return {"transcription": [await process_binary_file(file, uuid.uuid4()) for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/asr/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

async def process_binary_file(raw_binary_file: UploadFile, key: str):
    input_bytearray = await raw_binary_file.read()
    wav_file = convert_to_wav(input_bytearray, raw_binary_file.filename)
    data, sampling_rate = sf.read(wav_file)
    asr = AudioSpeechRecogniton()
    transcription = asr.transcribe(data)
    duration = librosa.get_duration(y=data, sr=sampling_rate)
    speech_details = {}
    speech_details['key'] = key
    speech_details['file_name'] = raw_binary_file.filename
    speech_details['duration'] = duration
    speech_details['transcription'] = transcription
    return json.dumps(speech_details)

def convert_to_wav(input_bytearray: bytes, filename: str, sampling_rate=16000):
    memory_file = io.BytesIO()
    memory_file.name = filename + ".wav"
    recording = np.frombuffer(input_bytearray, dtype=np.float32)
    sf.write(memory_file, recording, sampling_rate, subtype='PCM_24', format='wav')
    # reset internal stream to start
    memory_file.seek(0)
    return memory_file
    
def append_new_column(file_name):
    prefix = "cv-valid-dev/"
    suffix = ".mp3"
    new_file_name = prefix + file_name + suffix


class AudioSpeechRecogniton(object):

    def __init__(self) -> None:
        # which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single processor
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h", resume_download=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h", resume_download=True)
        
    def transcribe(self, speech):
        # retrieve input value
        input_values = self.processor(speech, return_tensors="pt").input_values  # Batch size 1
        # store non-normalized values
        logits = self.model(input_values).logits
        # retrieve prediction by passing logits to softmax
        prediction = torch.argmax(logits, dim = -1)
        # decode to obtain transcription
        transcription = self.processor.batch_decode(prediction)[0]
        return transcription
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # asr = AudioSpeechRecogniton()
    # input_bytearray = pickle.load(open("C:\\Users\\wyman\\Desktop\\xData\\asr\\processed_files\\sample-000001", "rb"))
    # wav_file = convert_to_wav(input_bytearray, "test")
    # data, sampling_rate = sf.read(wav_file)
    # asr = AudioSpeechRecogniton()
    # duration = librosa.get_duration(y=data, sr=sampling_rate)
    # transcription = asr.transcribe(data)
    # print(transcription)

