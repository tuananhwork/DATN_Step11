import os
import numpy as np
import webrtcvad
import noisereduce as nr
from pydub import AudioSegment
import librosa
from tqdm import tqdm

def process_audio_file(input_path, output_path):
    # Read and convert to mono, 16kHz
    audio = AudioSegment.from_wav(input_path).set_channels(1).set_frame_rate(16000)
    raw_audio = np.array(audio.get_array_of_samples())
    rate = audio.frame_rate

    # Noise reduction
    denoised_audio = nr.reduce_noise(y=raw_audio.astype(np.float32), sr=rate)

    # Ensure 16kHz sample rate
    if rate != 16000:
        raw_audio = librosa.resample(raw_audio.astype(np.float32), orig_sr=rate, target_sr=16000)
        denoised_audio = librosa.resample(denoised_audio.astype(np.float32), orig_sr=rate, target_sr=16000)
        rate = 16000

    # VAD processing
    vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
    frame_duration_ms = 30
    frame_length = int(rate * frame_duration_ms / 1000)
    frames = [denoised_audio[i:i+frame_length] for i in range(0, len(denoised_audio) - frame_length, frame_length)]

    def is_speech(frame):
        int16_frame = (frame * 32768).astype(np.int16)
        return vad.is_speech(int16_frame.tobytes(), rate)

    flags = [is_speech(frame) for frame in frames]
    speech_mask = np.repeat(flags, frame_length)
    speech_mask = np.pad(speech_mask, (0, len(denoised_audio) - len(speech_mask)), mode='constant')
    speech_audio = denoised_audio * speech_mask

    # Sliding window to find best 1.7s segment
    window_sec = 1.7
    window_len = int(window_sec * rate)
    stride = int(0.2 * rate)

    max_energy = 0
    best_segment = None

    for i in range(0, len(speech_audio) - window_len, stride):
        window = speech_audio[i:i+window_len]
        energy = np.sum(window.astype(np.float32)**2)
        if energy > max_energy:
            max_energy = energy
            best_segment = window

    # Trim 0.2s from start or end
    cut_sec = 0.2
    cut_len = int(cut_sec * rate)
    check_len = int(0.3 * rate)
    stride = int(0.01 * rate)

    max_energy = 0
    best_start = 0

    # Find 0.3s segment with highest energy
    for i in range(0, len(best_segment) - check_len + 1, stride):
        window = best_segment[i:i + check_len]
        energy = np.sum(window.astype(np.float32) ** 2)
        if energy > max_energy:
            max_energy = energy
            best_start = i

    best_start_sec = best_start / rate

    # Trim based on energy position
    if best_start_sec < 0.3:
        final_segment = best_segment[:-cut_len]
    else:
        if len(best_segment) > cut_len:
            final_segment = best_segment[cut_len:]
        else:
            final_segment = best_segment

    # Find best 0.8s segment and pad to 1s
    cad_len = int(0.8 * rate)
    stride = int(0.02 * rate)

    max_energy = 0
    best_start = 0

    for i in range(0, len(final_segment) - cad_len + 1, stride):
        window = final_segment[i:i + cad_len]
        energy = np.sum(window.astype(np.float32) ** 2)
        if energy > max_energy:
            max_energy = energy
            best_start = i

    cad_segment = final_segment[best_start:best_start + cad_len]

    # Pad to 1s
    pad_len = int(0.1 * rate)
    padded_segment = np.pad(cad_segment, (pad_len, pad_len), mode='constant')

    # Save processed audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_audio = AudioSegment(
        padded_segment.tobytes(), 
        frame_rate=rate,
        sample_width=2,  # 16-bit
        channels=1
    )
    processed_audio.export(output_path, format="wav")

def main():
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Process all WAV files in raw directory
    raw_dir = "data/raw"
    for command_dir in tqdm(os.listdir(raw_dir), desc="Processing commands"):
        command_path = os.path.join(raw_dir, command_dir)
        if not os.path.isdir(command_path):
            continue

        # Create corresponding directory in processed
        processed_command_dir = os.path.join("data/processed", command_dir)
        os.makedirs(processed_command_dir, exist_ok=True)

        # Process all WAV files in this command directory
        for wav_file in tqdm(os.listdir(command_path), desc=f"Processing {command_dir}", leave=False):
            if not wav_file.endswith('.wav'):
                continue

            input_path = os.path.join(command_path, wav_file)
            output_path = os.path.join(processed_command_dir, wav_file)
            
            try:
                process_audio_file(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    main() 