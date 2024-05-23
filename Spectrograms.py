import requests
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from MusicList import *
import json
import pandas as pd
import unidecode
import tensorflow_io as tfio
import soundfile as sf

# Descargar canciones y generar espectrogramas

all_df = pd.read_csv('Data/SpotifyData.csv', encoding='utf-8')

music_added = all_df[['track_name', 'artist', 'album', 'genre', 'track_link', 'track_preview']]

music_added = music_added.dropna()
print(music_added.head())
print(music_added.groupby('genre').track_preview.count())

for i in range(len(music_added)):
    url = music_added.iloc[i].track_preview

    track_name = unidecode.unidecode(music_added.iloc[i].track_name)
    track_name = track_name.replace(' ', '_')
    track_name = track_name.replace('/', '_')
    track_name = track_name.replace('?', '')
    track_name = track_name.replace('!', '')
    track_name = track_name.replace('¿', '')
    track_name = track_name.replace('¡', '')
    track_name = track_name.replace('"', '')
    track_name = track_name.replace('*', '')
    track_name = track_name.replace(':', '')

    path = f'Data/Samples/{music_added.iloc[i].genre}/{music_added.iloc[i].track_name.replace("/", "")}.mp3'
    path_img = f'Data/images_original/{music_added.iloc[i].genre}/{track_name}.png'
    #print("Cancion actual: ", music_added.iloc[i].track_name)

    # Codificar como utf-8
    path = path.encode('utf-8').decode('utf-8')
    path_img = path_img.encode('utf-8').decode('utf-8')

    if '"' in path:
        path = path.replace('"', '')
    if '?' in path:
        path = path.replace('?', '')
    if '*' in path:
        path = path.replace('*', '')
    if ':' in path:
        path = path.replace(':', '')

    if not os.path.exists(f'Data/Samples/{music_added.iloc[i].genre}'):
        os.makedirs(f'Data/Samples/{music_added.iloc[i].genre}')

    if not os.path.exists(f'Data/images_original/{music_added.iloc[i].genre}'):
        os.makedirs(f'Data/images_original/{music_added.iloc[i].genre}')

    # Descomentar para descargar canciones
    """download = requests.get(url)

    with open(path, 'wb') as f:
        f.write(download.content)
    """
    try:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        db_S = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))
        plt.savefig(path_img)
        plt.close()

        # Frequency Masking
        freq_mask = tfio.audio.freq_mask(S, param=10)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(freq_mask, ref=np.max))
        plt.savefig(path_img.replace('.png', '_freq_mask.png'))
        plt.close()

        # Time Masking
        time_mask = tfio.audio.time_mask(S, param=10)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(time_mask, ref=np.max))
        plt.savefig(path_img.replace('.png', '_time_mask.png'))
        plt.close()

        """# Frequency and Time Masking
        freq_time_mask = tfio.audio.time_mask(tfio.audio.freq_mask(S, param=10), param=10)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(freq_time_mask, ref=np.max))
        plt.savefig(path_img.replace('.png', '_freq_time_mask.png'))
        plt.close()"""

        # Pitch Shift
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
        S_shift = librosa.feature.melspectrogram(y=y_shift, sr=sr)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(S_shift, ref=np.max))
        plt.savefig(path_img.replace('.png', '_pitch_shift.png'))
        plt.close()

        """# Time Stretch
        y_time_stretch = librosa.effects.time_stretch(y, rate=2.0)
        S_time_stretch = librosa.feature.melspectrogram(y=y_time_stretch, sr=sr)
        plt.figure(figsize=(4.32, 2.88))
        librosa.display.specshow(librosa.amplitude_to_db(S_time_stretch, ref=np.max))
        plt.savefig(path_img.replace('.png', '_time_stretch.png'))
        plt.close()"""

    except (librosa.util.exceptions.ParameterError, ValueError) as e:
        print(f"Error con {music_added.iloc[i].track_name}")
        continue