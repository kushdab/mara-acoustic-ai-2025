# Mara Acoustic AI 2025

## Overview
Mara Acoustic AI is an edge-computing solution designed for conservancies to identify specific wildlife distress calls (e.g., elephants, rhinos) and illegal human activity (gunshots, chainsaws, vehicle engines) in real-time.

## Features
- **Edge Optimized:** Designed to run on Raspberry Pi or Jetson Nano using TFLite.
- **Real-time Analysis:** Continuous audio stream processing with low latency.
- **Acoustic Fingerprinting:** Uses MFCC feature extraction for robust identification.
- **Low Power:** Optimized for solar-powered remote deployments.

## Installation
1. Install system dependencies (for Linux):
   ```bash
   sudo apt-get install libportaudio2 libasound2-dev python3-pyaudio
   ```
2. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your trained TFLite model in the `models/` directory.
2. Run the processor:
   ```bash
   python audio_processor.py
   ```

## Project Structure
- `audio_processor.py`: Main logic for audio capture and inference.
- `models/`: Store TFLite models here.
- `requirements.txt`: Python dependencies.