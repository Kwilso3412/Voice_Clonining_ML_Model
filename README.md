# What Does the Model Do?
The model first uses the script 'Create_Voice_Profile' to train the model and save it in joblib file that can be used. The voice profile will then be loaded into the 'Read_Script' file and use this voice profile to and read any script with in the user defined folder. 

## What Technologies are Used?
- Python
- Tortoise TTS
- Pytorch
- Conda (optional)

## Do I need a Virtual Environment?
I highly recommend using as virtual environment because Tortoise TTS was desgined for Python version 3.10. If you have a higher version of python then you will have some trouble running the scripts. 

### Okay got it what do I need to pip install on it?
- tqdm
- joblib
- numpy
- soundfile
- pydub
- docx2txt
- torch
- torchaudio --index-url https://download.pytorch.org/whl/cu121
  * For GPU support (replace the cu121 with your version of cuda)

## Are there any Specific Requirements for Optimal Voice Cloning?
Yes for optimal training Tortoise TTS has these specific requirements:
### Sample Count

- **Minimum**: 2-3 samples
- **Recommended**: 5-7 samples
- **Optimal**: 8-10 samples

### Audio Specifications

- **Total duration**: 1-2 minutes of speech
- **Individual clips**: 6-10 seconds each
- **Format**: WAV files (44.1kHz or 48kHz)
- **Quality**: High clarity, minimal background noise

### Content Requirements

- Include varied intonation patterns
- Mix statements and questions
- Use phonetically diverse sentences

*Its important to note that Tortoise TTS work*
- *Tortoise TTS works extremely well with consitant audio quailty across samples*
- *Tortoise TTS tends to perform bettter with longer longer samples than other TTS systems*
- *The diffusion model approach requires fewer samples but benefits significantly from sample diversity.* 

## I want to make adjustments and fine tune my voice?
The code in the function 'generate_audiobook_cuda_optimized' in the 'Read_Script' file is currently set to random so the speaker will be random whne it reads the script. However this can be fine tuned and changed so that it sounds more natural and closer to the user speaking. 

### Metrics to change:
- Low temperature (0.3-0.4) = minimal voice variation
- Fewer autoregressive samples = less opportunity for voice drift
- High cond_free_k (4.0-5.0) = forces adherence to your voice profile
- Fixed random seeds = reproducible results

