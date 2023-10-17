import pathlib
from pathlib import Path
import os
import pickle
import librosa
import traceback

class preprocessRawAudio(object):
    def __init__(self, input_directory, output_directory) -> None:
        self.input_directory = input_directory
        self.output_directory = output_directory

    def process_directory(self, target_sampling = 16000):
        list_file_path = [p for p in pathlib.Path(self.input_directory).iterdir() if p.is_file()]
        if len(list_file_path) == 0:
            print("No files to process")
        else:
            for filename in list_file_path:
                # sound wave represented as 1D numpy floating point array
                sound_wave, sampling_rate = librosa.load(filename)
                audio_data = self.sampling_audio(sound_wave, sampling_rate, target_sampling)
                filename_no_suffix = str(filename).split('\\')[-1].split('.mp3')[0]
                self.convertToBinary(filename_no_suffix, audio_data)
                
        
    def sampling_audio(self, sound_wave, sampling_rate, target_sampling: int):
        audio_data = sound_wave
        if sampling_rate != target_sampling:
            audio_data = librosa.resample(sound_wave, orig_sr=sampling_rate, target_sr=target_sampling)
        return audio_data

    def convertToBinary(self, filename_no_suffix, audio_data):
        binary_string = audio_data.tobytes()
        saved_path = os.path.join(self.output_directory, "processed_files", filename_no_suffix)
        with open(saved_path, "wb") as savedBytesFile:
            pickle.dump(binary_string, savedBytesFile)
        print(filename_no_suffix + " saved to " + saved_path)
    
if __name__ == "__main__":
    input_directory = "C:\\Users\\wyman\\OneDrive\\Desktop\\cv-valid-dev\\cv-valid-dev"
    destination_directory = "C:\\Users\\wyman\\Desktop\\xData\\asr"
    done_file = Path(os.path.join(destination_directory, "processed_files", "done.txt"))
    processAudio = preprocessRawAudio(input_directory, destination_directory)
    if not done_file.is_file():
        # file does not exist
        try:
            # create directory called processed_files if it does not exist
            Path(os.path.join(destination_directory, "processed_files")).mkdir(parents=True, exist_ok=True)
            processAudio.process_directory()
            with open(os.path.join(destination_directory, "processed_files", "done.txt"), "w") as file:
                file.write("Finished converting audio files to binary")
        except IOError:
            print("Failed to process audio files. Exception: {}", traceback.format_exc())
    else:
        print("done")