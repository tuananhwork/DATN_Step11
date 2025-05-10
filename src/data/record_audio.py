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

# Danh sách các lệnh
COMMANDS = {
    '1': 'bat_den',
    '2': 'tat_den',
    '3': 'nhiet_do',
    '4': 'do_am',
    '5': 'bat_quat',
    '6': 'tat_quat',
    '7': 'mo_rem',
    '8': 'dong_rem',
    '9': 'bat_tv',
    '10': 'tat_tv',
    '12': 'bat_dieu_hoa',
    '13': 'tat_dieu_hoa'
}

def get_next_recording_id(command_dir):
    """Lấy ID tiếp theo cho bản ghi mới"""
    existing_files = [f for f in os.listdir(command_dir) if f.endswith('.wav')]
    if not existing_files:
        return 1
    max_id = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files])
    return max_id + 1

def record_audio(duration=3, sample_rate=SAMPLE_RATE):
    """
    Ghi âm với thời gian cố định
    
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
    print(f"Bắt đầu ghi âm trong {duration} giây...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()
    print("Ghi âm hoàn tất!")
    return audio_data

def save_audio(audio_data, output_dir, command, speaker_id, recording_id, sample_rate=SAMPLE_RATE):
    """
    Lưu file âm thanh với tên theo format: command_speaker[xx]_xxx.wav
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        Dữ liệu âm thanh
    output_dir : str
        Thư mục lưu file
    command : str
        Tên lệnh
    speaker_id : str
        ID người nói (2 chữ số)
    recording_id : int
        Số thứ tự bản ghi
    sample_rate : int
        Tần số lấy mẫu
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách file của người nói hiện tại
    existing_files = [f for f in os.listdir(output_dir) 
                     if f.endswith('.wav') and f'speaker{speaker_id}' in f]
    
    # Nếu chưa có file nào của người này, bắt đầu từ 001
    if not existing_files:
        next_id = 1
    else:
        # Lấy số thứ tự lớn nhất của người này
        max_id = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files])
        next_id = max_id + 1
    
    # Tạo tên file với format mới
    recording_id_str = f"{next_id:03d}"
    filename = f"{command}_speaker{speaker_id}_{recording_id_str}.wav"
    output_path = os.path.join(output_dir, filename)
    
    # Lưu file
    sf.write(output_path, audio_data, sample_rate)
    print(f"Đã lưu file: {output_path}")

def main():
    # Tạo thư mục data/raw nếu chưa tồn tại
    base_dir = os.path.join('data', 'raw')
    os.makedirs(base_dir, exist_ok=True)

    # Lưu trữ lựa chọn trước đó
    last_command = None
    last_speaker_id = None

    while True:
        print("\n=== CHƯƠNG TRÌNH GHI ÂM LỆNH ĐIỀU KHIỂN ===")
        print("\nDanh sách lệnh:")
        for key, value in COMMANDS.items():
            print(f"{key}. {value}")

        # Chọn lệnh
        command_prompt = "\nChọn số thứ tự lệnh (hoặc 'q' để thoát)"
        if last_command:
            command_prompt += f" [mặc định: {last_command}]"
        command_prompt += ": "
        
        command_choice = input(command_prompt).strip()
        if command_choice.lower() == 'q':
            break

        if not command_choice and last_command:
            command_choice = last_command
        elif command_choice not in COMMANDS:
            print("Lựa chọn không hợp lệ!")
            continue

        command = COMMANDS[command_choice]
        last_command = command_choice
        command_dir = os.path.join(base_dir, command)
        os.makedirs(command_dir, exist_ok=True)

        # Nhập ID người nói
        speaker_prompt = "Nhập ID người nói (2 chữ số, ví dụ: 01)"
        if last_speaker_id:
            speaker_prompt += f" [mặc định: {last_speaker_id}]"
        speaker_prompt += ": "

        while True:
            speaker_id = input(speaker_prompt).strip()
            if not speaker_id and last_speaker_id:
                speaker_id = last_speaker_id
                break
            if len(speaker_id) == 2 and speaker_id.isdigit():
                last_speaker_id = speaker_id
                break
            print("ID người nói phải có 2 chữ số!")

        # Lấy ID bản ghi tiếp theo
        recording_id = get_next_recording_id(command_dir)
        recording_id_str = f"{recording_id:03d}"

        # Tạo tên file
        filename = f"{command}_speaker{speaker_id}_{recording_id_str}.wav"
        filepath = os.path.join(command_dir, filename)

        # Ghi âm
        audio_data = record_audio()
        
        # Lưu file
        save_audio(audio_data, command_dir, command, speaker_id, recording_id)

        # Hỏi người dùng có muốn ghi âm tiếp không
        if input("\nBạn có muốn ghi âm tiếp không? [Y/n]: ").lower() == 'n':
            break

if __name__ == "__main__":
    main() 