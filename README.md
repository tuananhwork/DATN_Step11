# Audio Command Detection Project

## Project Structure

```
.
├── data/
│   ├── processed/          # Processed audio files
│   ├── raw/               # Raw audio files
│   └── models/            # Trained model files
├── notebooks/
│   ├── 1.0_preprocess.ipynb           # Initial data preprocessing
│   ├── 2.0_batch_preprocess.ipynb     # Batch preprocessing of audio files
│   ├── 3.0_mel_features_extraction.ipynb  # Feature extraction from audio
│   ├── 4.0_train_model.ipynb          # Model training
│   ├── 5.0_validate.ipynb             # Model validation
│   ├── 6.0_convmixer.ipynb            # ConvMixer model training
│   └── 7.0_convmixer_validate.ipynb   # ConvMixer model validation
├── src/
│   └── audio_command_detector.py      # Main application script
└── requirements.txt                   # Project dependencies
```

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Direct Execution

To run the audio command detector directly:

```bash
python src/audio_command_detector.py
```

This will:

- Start the audio command detection system
- Press Enter to start recording
- The system will process the audio and predict the command
- Results will show the predicted command and confidence scores
- Press Ctrl+C to exit

### Option 2: Model Experimentation

#### Path 1: Standard Model

1. Run the notebooks in sequence:
   ```bash
   jupyter notebook notebooks/2.0_batch_preprocess.ipynb
   jupyter notebook notebooks/3.0_mel_features_extraction.ipynb
   jupyter notebook notebooks/4.0_train_model.ipynb
   jupyter notebook notebooks/5.0_validate.ipynb
   ```

#### Path 2: ConvMixer Model

1. Run the notebooks in sequence:
   ```bash
   jupyter notebook notebooks/2.0_batch_preprocess.ipynb
   jupyter notebook notebooks/3.0_mel_features_extraction.ipynb
   jupyter notebook notebooks/6.0_convmixer.ipynb
   jupyter notebook notebooks/7.0_convmixer_validate.ipynb
   ```

## Notes

- Make sure you have all required dependencies installed
- The model expects audio files in WAV format
- The system is optimized for 16kHz mono audio
- Supported commands include: bat_den, bat_dieu_hoa, bat_quat, bat_tv, do_am, dong_rem, mo_rem, nhiet_do, tat_den, tat_dieu_hoa, tat_quat, tat_tv
