import numpy as np
import sounddevice as sd
import librosa
import time
import logging
import tflite_runtime.interpreter as tflite
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaraAcousticAI:
    def __init__(self, model_path="models/classifier.tflite", threshold=0.8):
        self.rate = 22050  # Sample rate
        self.duration = 3  # Seconds to analyze per window
        self.threshold = threshold
        self.chunk_size = self.rate * self.duration
        
        # Initialize TFLite interpreter
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logging.info("Edge model loaded successfully.")
        except Exception as e:
            logging.error(f"Model load failed: {e}. Running in dummy mode.")
            self.interpreter = None

    def preprocess(self, audio_data):
        # Convert to MFCC (Feature extraction)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled.reshape(1, -1).astype(np.float32)

    def run_inference(self, features):
        if not self.interpreter:
            return np.random.random() # Simulate for demo
        
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        return prediction[0][0]

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Status: {status}")
        
        # Flatten the buffer
        audio_signal = indata[:, 0]
        features = self.preprocess(audio_signal)
        score = self.run_inference(features)

        if score > self.threshold:
            self.trigger_alert(score)

    def trigger_alert(self, score):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.warning(f"[ALERT] Possible Threat Detected! Confidence: {score:.2f} at {timestamp}")
        # Integration point for LoRaWAN or SMS gateway would go here

    def start_listening(self):
        logging.info("Starting real-time acoustic monitoring...")
        with sd.InputStream(channels=1, samplerate=self.rate, 
                          blocksize=self.chunk_size, callback=self.audio_callback):
            while True:
                time.sleep(0.1)

if __name__ == "__main__":
    detector = MaraAcousticAI()
    try:
        detector.start_listening()
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")