import whisper
import os
import time
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch

def detect_chunks(audio_segment, min_silence_len=2000, silence_thresh=-40):
    """Detect non-silent chunks in an audio segment."""
    non_silent_ranges = detect_nonsilent(audio_segment, 
                                         min_silence_len=min_silence_len, 
                                         silence_thresh=silence_thresh)
    chunks = [audio_segment[start:end] for start, end in non_silent_ranges]
    return chunks

def transcribe_audio(file_path, model_name="base"):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = whisper.load_model(model_name).to(device)
    print(f"Transcribing {file_path}...")
    
    # Load audio file
    audio = AudioSegment.from_file(file_path, format="mp3")
    
    # Detect chunks
    chunks = detect_chunks(audio)
    print(f"Detected {len(chunks)} non-silent chunks")
    
    full_transcription = ""
    
    for i, chunk in enumerate(chunks):
        # Export chunk to a temporary file
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        # Transcribe the chunk
        result = model.transcribe(chunk_path)
        chunk_text = result["text"]
        
        print(f"{i+1}. {chunk_text}")
        full_transcription += chunk_text + " "
        
        # Remove temporary file
        os.remove(chunk_path)
    
    detected_language = result["language"]  # Using the language from the last chunk
    
    return full_transcription.strip(), detected_language

def format_elapsed_time(seconds):
    """Format elapsed time into hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s" if hours else f"{minutes}m {secs:.2f}s"

def main():
    start_time = time.time()  # Start timer for the entire script
    
    audio_file = "./mp3s/Anger Management.mp3"
    
    if not os.path.exists(audio_file):
        print(f"Error: The file {audio_file} does not exist.")
        return
    
    model_size = "small"
    
    transcription, language = transcribe_audio(audio_file, model_size)
    
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(f"Detected language: {language}\n\n")
        f.write(transcription)
    
    print("\nTranscription saved to transcription.txt")
    
    # Calculate and display the total elapsed time
    total_elapsed_time = time.time() - start_time
    formatted_time = format_elapsed_time(total_elapsed_time)
    print(f"Total elapsed time: {formatted_time}")

if __name__ == "__main__":
    main()