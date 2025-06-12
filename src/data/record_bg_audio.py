import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from pathlib import Path

# Cấu hình ghi âm
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # Mono
DTYPE = np.int16     # 16-bit

def record_background_noise(duration=60, sample_rate=SAMPLE_RATE):
    """
    Ghi âm tiếng ồn nền với thời gian cố định
    
    Parameters:
    -----------
    duration : float
        Thời gian ghi âm (giây)
    sample_rate : int
        Tần số lấy mẫu
    
    Returns:
    --------
    audio_data : numpy.ndarray
        Dữ liệu âm thanh đã ghi
    """
    print(f"Bắt đầu ghi âm tiếng ồn nền trong {duration} giây...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()
    print("Ghi âm hoàn tất!")
    return audio_data

def save_background_noise(audio_data, output_dir, sample_rate=SAMPLE_RATE):
    """
    Lưu file âm thanh tiếng ồn nền
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        Dữ liệu âm thanh
    output_dir : str
        Thư mục lưu file
    sample_rate : int
        Tần số lấy mẫu
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bg_noise_{timestamp}.wav"
    output_path = os.path.join(output_dir, filename)
    
    # Lưu file
    sf.write(output_path, audio_data, sample_rate)
    print(f"Đã lưu file: {output_path}")

def main():
    # Lấy đường dẫn hiện tại và tạo thư mục data/bg_noise
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(current_dir, 'data', 'bg_noise')
    
    print("\n=== CHƯƠNG TRÌNH GHI ÂM TIẾNG ỒN NỀN ===")
    print(f"File sẽ được lưu tại: {output_dir}")
    
    # Ghi âm
    audio_data = record_background_noise()
    
    # Lưu file
    save_background_noise(audio_data, output_dir)
    
    print("\nHoàn tất ghi âm tiếng ồn nền!")

if __name__ == "__main__":
    main()
