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
Parameter Ranges for Voice Consistency:

<ins>Temperature Range</ins>:

- 0.1-0.3: Robotic but identical voice across chunks
- 0.4-0.6: Balanced consistency with some naturalness
- 0.7-0.9: Natural but voice may drift between chunks
- 1.0-1.2: Very natural but high chance of voice changes
Effect: Lower = same voice, higher = more natural but inconsistent

<ins>Autoregressive Samples Range</ins>:

- 4-8: Very consistent, lower quality speech
- 12-16: Good consistency, decent quality
- 32-48: Moderate consistency, good quality
- 64-128: Poor consistency, best quality
Effect: Fewer samples = same voice, more samples = better pronunciation but voice drift

<ins>cond_free_k Range</ins>:

- 0.5-1.0: Barely uses your voice profile (generic voices)
- 1.5-2.5: Standard voice conditioning (original settings)
- 3.0-4.0: Strong voice conditioning (recommended for consistency)
- 4.5-6.0: Maximum voice conditioning (may sound forced)
Effect: Higher = forces your exact voice characteristics

<ins>top_p Range</ins>:

- 0.3-0.5: Very limited word choices, consistent voice
- 0.6-0.8: Balanced word variety and consistency
- 0.9-0.95: High word variety, more voice variation
Effect: Lower = same speaking patterns, higher = more natural variety

<ins>Some Combinations</ins>:
* **Maximum Consistency (Robotic)**:
  - temperature=0.2, samples=8, cond_free_k=5.0, top_p=0.4

* **Balanced (Recommended)**:
  - temperature=0.4, samples=12, cond_free_k=4.0, top_p=0.6

* **Natural but Risky**:
  - temperature=0.7, samples=32, cond_free_k=2.5, top_p=0.8

* **Key Trade-off**:
  - Consistency vs Natural Sound - Lower values = identical voice but more robotic speech

## Diffusion Iterations - Audio Quality Control:
What It Does:
- Refines raw audio through multiple denoising steps
- Starts with noise → gradually creates clean speech
- Each iteration = one step of audio improvement
- More iterations = cleaner, higher quality audio

<ins>Iteration Ranges:
- 10-25: Very fast, rough/fuzzy audio quality
- 25-50: Balanced speed/quality, decent clarity
- 50-100: Good quality, clear speech
- 100-200: High quality, very clean audio
- 200+: Diminishing returns, extremely slow

### Effects on Output:

<ins>Low iterations (10-25)</ins>:
- Fast generation
- Fuzzy/distorted audio
- Background noise/artifacts


<ins>Medium iterations (50-100)</ins>:
- Moderate speed
- Clear, usable speech
- Minimal artifacts


<ins>High iterations (100+)</ins>:
- Slow generation
- Crystal clear audio
- No background noise

<ins>Voice Consistency Impact</ins>:
- Does NOT affect voice consistency - that's controlled by temperature/samples
- Only affects audio clarity - same voice, just cleaner/fuzzier
- Can run low iterations for consistency testing, then increase for final quality

<ins>Recommendeds</ins>:
- Testing: 25 iterations (fast, see if voice is consistent)
- Production: 50-100 iterations (good quality without excessive time)
- Final/Important: 100+ iterations (maximum clarity)
