# Funcionamiento:
* MusicList.py contiene la información de las playlists de Spotify utilizadas
* GetSpotifyData.py crear el archivo SpotifyData.csv correspondiente a las canciones obtenidas de las playlists
* Spectrograms.py descarga las muestras de SpotifyData.csv (Al descomentar la línea de código pertinente) y crea los escpectrogramas
* NeuroNet.py crea y realiza el entrenamiento de una red neuronal en base a los espectrogramas en images_original
* ModelTests.py son las pruebas del modelo creado por NeuroNet.py, si el modelo no se puede cargar asegurarse de que la versión de Tensorflow sea la misma que con la que se creo el modelo
* ImagenRed.py genera un pdf de la red utilizando [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
