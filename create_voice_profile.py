"""
How To Use

1) Put all your voice samples in the ./voice_samples folder
2) Run the script with no arguments: python create_voice_profile.py
3) All audio files will be processed into a single voice profile

- Automatically processes all voice samples from a specific folder
- Requires at least 10 seconds of clear speech
- Better results with 30-60 seconds total across 5-10 samples

I used 50 after that you will get diminishing returns 
"""

import os
import sys
import subprocess
import joblib
import soundfile as sf
import numpy as np
import torch
import glob
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio

"""
This function will take the voice recording and then get it ready for the cloning system. It converts
the audio to mono (one channel instead of stereo), it will change the sample rate to 22.05kHz and save it as a new file
"""
def prepare_audio(audio_path, output_path=None):
    """
    Process audio files to ensure it's compatible with Tortoise TTS.
    Converts to mono, 22.05kHz, normalized.
    
    Args:
        audio_path: Path to the input file 
        output_path: Path to save the processed audio (optional)

    Returns:
        Path to the processed audio file
    """
    # Default output path if none provided 
    if output_path is None:
        # Get the directory where the original file is located
        file_dir = os.path.dirname(audio_path)
        
        # Create a 'processed' folder in that same directory
        processed_dir = os.path.join(file_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get just the filename
        base_filename = os.path.basename(audio_path)
        filename, ext = os.path.splitext(base_filename)
        
        # Build the output path
        output_path = os.path.join(processed_dir, f"{filename}_processed.wav")
    
    # Load the audio
    print(f"Loading audio from {audio_path}...")
    try:
        audio_data, sample_rate = sf.read(audio_path)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Resample to 22.05kHz if needed (what Tortoise expects)
    if sample_rate != 22050:
        print(f"Resampling from {sample_rate}Hz to 22050Hz..")
        sf.write(output_path, audio_data, 22050)
    else:
        sf.write(output_path, audio_data, sample_rate)
    
    print(f"Audio processed and saved to {output_path}")
    return output_path 


"""
This function will analyze your voice recording to capture your unique voice characteristics. It extracts information 
about how you sound and saves it as a "voice profile" that can be used later to generate speech that sounds like you.
"""
def create_voice_profile(audio_paths, profile_name, output_dir="voice_profiles"):
    """
    Create a voice profile by extracting conditioning latents from audio samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process audio samples
    processed_audio_paths = []
    print(f"Processing {len(audio_paths)} audio files...")
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Processing file {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
        processed_path = prepare_audio(audio_path)
        if processed_path:
            processed_audio_paths.append(processed_path)
    
    if not processed_audio_paths:
        print("No valid audio files were processed.")
        return None
    
    # Initialize Tortoise (no training parameters)
    print("Initializing Tortoise TTS...")
    tts = TextToSpeech()
    
    # Load audio samples
    print("Loading audio samples...")
    voice_samples = []
    for path in processed_audio_paths:
        audio = load_audio(path, 22050)
        voice_samples.append(audio)
    
    # Extract conditioning latents (this is what Tortoise uses for voice cloning)
    print("Extracting voice conditioning latents...")
    conditioning_latents = tts.get_conditioning_latents(voice_samples)
    
    # Save the voice profile
    profile_dir = os.path.join(output_dir, profile_name)
    os.makedirs(profile_dir, exist_ok=True)
    
    voice_data = {
        'name': profile_name,
        'conditioning_latents': conditioning_latents,
        'sample_paths': processed_audio_paths,
        'created_at': np.datetime64('now'),
        'sample_count': len(voice_samples)
    }
    
    profile_path = os.path.join(profile_dir, f"{profile_name}_profile.joblib")
    joblib.dump(voice_data, profile_path)
    
    print(f"Voice profile '{profile_name}' created from {len(processed_audio_paths)} samples!")
    return profile_path

def main():
    """
    Main function to create voice profiles from all audio files in a specific folder
    """
    print("Starting to create voice profile")
    
    # Set the folder paths directly here
    samples_folder = "./voice_samples/me"
    output_dir = "./voice_profiles"
    train_iterations = 750
    
    # Set the voice profile name
    profile_name = "my_voice"
    
    # Get all audio files from the folder
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(samples_folder, f"*{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {samples_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files in {samples_folder}")
    
    # Check if we have enough samples
    if len(audio_files) < 5:
        print("WARNING: For best results, at least 5 samples are recommended")
    elif len(audio_files) > 50:
        print("NOTE: Processing a large number of samples (>50) may take significant time")
        print("Would you like to proceed with all samples? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Please reduce the number of samples and try again.")
            return
    
    # Print the files found
    print("Processing these audio files:")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Estimate processing time
    sample_count = len(audio_files)
    est_minutes = max(5, (sample_count * 1.5) if torch.cuda.is_available() else (sample_count * 5))
    print(f"Estimated processing time: ~{est_minutes:.1f} minutes")
    
    # Create the voice profile
    profile_path = create_voice_profile(
        audio_files, 
        profile_name,
        output_dir
    )
    
    if profile_path:
        print("Voice Profile complete!")
        print(f"Successfully processed {sample_count} samples")
    else:
        print("Failed to create voice profile")

if __name__ == "__main__":
    main()