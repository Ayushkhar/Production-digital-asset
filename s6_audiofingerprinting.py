import librosa
import numpy as np

def extract_features(video_path):
    y,sr=librosa.load(video_path,sr=22050)

    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    mfcc_mean =np.mean(mfcc,axis=1)
    return mfcc_mean.astype('float32')

orig_audio=extract_features("original.mp4")
pirated_audio=extract_features("pirated_crop.mp4")
def cosine_sim(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_sim(orig_audio, pirated_audio)
print("Audio similarity:",similarity)

if similarity > 0.85:
    print("AUDIO: PIRATED")
else:
    print("AUDIO: NOT PIRATED")