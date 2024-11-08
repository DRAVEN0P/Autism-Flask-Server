import numpy as np
import librosa
import parselmouth
import statistics
from parselmouth.praat import call



def Feature(path, f0min, f0max,unit,gender,age):
    print()
    # Load the sound using parselmouth
    sound = parselmouth.Sound(path)

    # Duration of the sound
    duration = call(sound, "Get total duration")

    # Calculate F0 (mean)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)

    # Harmonicity (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Point process for jitter and shimmer
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    f5_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f5 = call(formants, "Get value at time", 5, t, 'Hertz', 'Linear')
        if f1: f1_list.append(f1)
        if f2: f2_list.append(f2)
        if f3: f3_list.append(f3)
        if f4: f4_list.append(f4)
        if f5: f5_list.append(f5)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    f5_list = [f5 for f5 in f5_list if str(f5) != 'nan']

    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list) if f1_list else float('nan')
    f2_mean = statistics.mean(f2_list) if f2_list else float('nan')
    f3_mean = statistics.mean(f3_list) if f3_list else float('nan')
    f4_mean = statistics.mean(f4_list) if f4_list else float('nan')
    f5_mean = statistics.mean(f5_list) if f5_list else float('nan')

    y, sr = librosa.load(path, sr=None)

    # Calculate Short-term Energy (SoE)
    frame_size = int(0.025 * sr)  # 25ms frame size
    hop_length = int(0.01 * sr)  # 10ms hop length

    # Calculate mean energy (RMS)
    rms = librosa.feature.rms(y=y)
    mean_energy = np.mean(rms)

    # Calculate short-term energy (SoE) using Hamming window
    frame_size = int(0.025 * sr)  # 25ms frame size
    hop_length = int(0.01 * sr)  # 10ms hop length
    window = np.hamming(frame_size)
    soe = [
        sum((y[i:i + frame_size] * window) ** 2)
        for i in range(0, len(y) - frame_size + 1, hop_length)
    ]
    mean_soe = np.mean(soe)

    # Calculate zero-crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_size, hop_length=hop_length)
    mean_zcr = np.mean(zcr)

    return duration, meanF0, hnr, localJitter, localShimmer, mean_energy, mean_soe, mean_zcr,f1_mean, f2_mean, f3_mean, f4_mean, f5_mean,


# Feature("/Users/draven/Downloads/Sample 7 F46.wav",75, 300, "Hertz",'F',35)


