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

ids = json.load(open('IDs.json'))

# Recuperar info API Spotify
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
                                                                                client_id = ids['Client_ID'],
                                                                                client_secret = ids['Client_Secret']
                                                                                ))
def get_audio_features(playlists, genre):

    features = ["track_name", "artist", "genre", "track_uri", "track_link", "track_preview", "album"]

    genre_df = pd.DataFrame(columns=features)

    for playlist in playlists:

        try:
            playlist_info = spotify.playlist_items(playlist)['items']
        except:
            print(f"Error with playlist {playlist}")
            exit(1)

        for track in playlist_info:

            track_info = {}

            track_info['track_name'] = track['track']['name']
            track_info['artist'] = track['track']['artists'][0]['name']
            track_info['track_uri'] = track['track']['uri']
            track_info['track_link'] = track['track']['external_urls']['spotify']
            track_info['track_preview'] = track['track']['preview_url']
            track_info['genre'] = genre
            track_info['album'] = track['track']['album']['name']

            track_df = pd.DataFrame(track_info, index=[0])
            genre_df = pd.concat([genre_df, track_df], ignore_index=True)

    genre_df.drop_duplicates(subset=['track_uri'], inplace=True)

    return genre_df

rock_df = get_audio_features(rock, 'Rock')
pop_df = get_audio_features(pop, 'Pop')
hiphop_df = get_audio_features(hiphop, 'Hip Hop')
rap_df = get_audio_features(rap, 'Rap')
country_df = get_audio_features(country, 'Country')
latin_df = get_audio_features(latin, 'Latin')
kpop_df = get_audio_features(kpop, 'K-Pop')
classical_df = get_audio_features(classical, 'Classical')
jazz_df = get_audio_features(jazz, 'Jazz')
indie_df = get_audio_features(indie, 'Indie')
metal_df = get_audio_features(metal, 'Metal')
edm_df = get_audio_features(EDM, 'EDM')

all_df = pd.concat([rock_df, pop_df, hiphop_df, rap_df, country_df, latin_df, kpop_df, classical_df, jazz_df, indie_df, metal_df, edm_df], ignore_index=True)

if not os.path.exists('Data'):
    os.makedirs('Data')

all_df.to_csv('Data/SpotifyData.csv', index=False)
