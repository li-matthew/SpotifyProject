import spotipy
import spotipy.util as util
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np
import seaborn as sn

#Get Songs
token = util.prompt_for_user_token('1256293535','playlist-modify-public',client_id='2dceb191a60046449db76d84e7d424c1',client_secret='55303fe8baf84690a75afa5f37ef311a',redirect_uri='https://www.google.com/')

spotify = spotipy.Spotify(token)

def create(items, playlist):
    total = []
    for song in items:
        temp = []
        if song.get('track') != None:
            id = song.get('track').get('id')
            features = spotify.audio_features([id])[0]
            if features != None:
                temp.append(song.get('track').get('artists')[0].get('name').encode('ascii', 'ignore'))
                temp.append(song.get('track').get('name').encode('ascii', 'ignore'))
                temp.append(id)
                temp.append(playlist)
                for y in ['energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'danceability', 'key', 'duration_ms', 'loudness', 'valence', 'mode']:
                    temp.append(features.get(y))
                total.append(temp)
                print(temp)
    return total

def get_playlist_tracks(username,playlist_id, playlist):
    results = spotify.user_playlist_tracks(username,playlist_id)
    songs = create(results.get('items'), playlist)
    while results['next']:
        results = spotify.next(results)
        for x in create(results.get('items'), playlist):
            songs.append(x)
    return songs

#My Songs
def get_my_songs():
    vybes = get_playlist_tracks('1256293535', '4QklqO91zFyLHaZGug6AQT', 'vybes')
    chillz = get_playlist_tracks('1256293535', '2bDb7JITikmUGM5QooCTfB', 'chillz')
    gibberish = get_playlist_tracks('1256293535', '4IRBP431dMVa04bnjG88VY', 'gibberish')
    wack = get_playlist_tracks('1256293535', '56iN0HPEq6nl2lLYQlSch5', 'wack')
    singed = get_playlist_tracks('1256293535', '3QmptCYgBwtHYbcxtvXJco', 'singed')
    litt = get_playlist_tracks('1256293535', '3nF16flWwFdkDU8yUtbXQp', 'litt')
    bars = get_playlist_tracks('1256293535', '6wM0xkWBbSA9DIhXXNN6JB', 'bars')
    kms = get_playlist_tracks('1256293535', '3bOmC7EfH5I0ESVztnKgTZ', 'kms')
    total = vybes + chillz + gibberish + wack + singed + litt + bars + kms
    data = pd.DataFrame(total,
                        columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness',
                                 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness',
                                 'Valence', 'Mode'])
    data = data.drop_duplicates(subset=['Artist', 'Name'], keep='first')
    data.to_csv('/Users/matthewli/Documents/Spotify Project/Spotify.csv')

#Extra Data
def get_top_songs():
    mytop = []
    my2016 = get_playlist_tracks('1256293535', '37i9dQZF1CyWPoeKJeEUem', 'my2016')
    mytop = mytop + my2016
    my2016 = pd.DataFrame(my2016, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    my2016.to_csv('/Users/matthewli/Documents/Spotify Project/My2016.csv')

    my2017 = get_playlist_tracks('1256293535', '37i9dQZF1E9QuqoF4pvCjO', 'my2017')
    mytop = mytop + my2017
    my2017 = pd.DataFrame(my2017, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    my2017.to_csv('/Users/matthewli/Documents/Spotify Project/My2017.csv')

    my2018 = get_playlist_tracks('1256293535', '37i9dQZF1EjofvKv8uxeuM', 'my2018')
    mytop = mytop + my2018
    my2018 = pd.DataFrame(my2018, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    my2018.to_csv('/Users/matthewli/Documents/Spotify Project/My2018.csv')

    my2019 = get_playlist_tracks('1256293535', '37i9dQZF1Et8A31ON5uKGj', 'my2019')
    mytop = mytop + my2019
    my2019 = pd.DataFrame(my2019, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    my2019.to_csv('/Users/matthewli/Documents/Spotify Project/My2019.csv')

    mytop = pd.DataFrame(mytop, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability', 'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    mytop = mytop.drop_duplicates(subset=['Artist', 'Name'], keep='first')
    mytop.to_csv('/Users/matthewli/Documents/Spotify Project/MyTop.csv')

#Read Data
def read_data(file):
    dataset = pd.read_csv(file)
    dataset = dataset.drop(dataset.columns[0], axis=1)
    transform = dataset.drop(dataset.columns[0:4], axis=1)
    for x in ['Duration', 'Tempo', 'Key', 'Loudness', 'Instrumentalness', 'Mode', 'Liveness']:
        del transform[x]
    transform = transform.astype('float')
    return transform

#Inertia
def get_inertia(data):
    clusters = [1,2,3,4,5,6,7,8,9,10]
    inertia = []
    for x in clusters:
        kmeans = KMeans(n_clusters=x)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    plt.scatter(clusters, inertia)
    plt.savefig('/Users/matthewli/Documents/Spotify Project/inertia.png')
    plt.clf()

#PCA
def pca(data):
    transform = preprocessing.scale(data, with_std=False)
    pca = decomposition.PCA(n_components = 3)
    pca.fit(transform)
    pcatransform = pca.transform(transform)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pcatransform)
    pcacenters = kmeans.cluster_centers_
    pcalabels = kmeans.labels_
    print(pca.explained_variance_ratio_)
    pcay_kmeans = kmeans.predict(pcatransform)
    plt.scatter(pcatransform[:,0],pcatransform[:,1], c=pcay_kmeans, alpha=0.25)
    plt.scatter(pcacenters[:, 0], pcacenters[:, 1], c='black', s=200, alpha=0.5);
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig('/Users/matthewli/Documents/Spotify Project/pca.png')
    data['Clusters'] = pcalabels
    data.to_csv('/Users/matthewli/Documents/Spotify Project/Spotify_PCA.csv')
    plt.clf()

#Predict
def knn(data):
    x = data.iloc[:, :-1]
    print(x)
    y = data.iloc[:, 5]
    print(y)
    x = x.astype('float')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    #Best K value
    k_range = range(1,30)
    error = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_test)

        error.append(np.mean(y_pred != y_test))
    plt.scatter(k_range, error)
    plt.plot(k_range, error)
    plt.savefig('/Users/matthewli/Documents/Spotify Project/KNN_Error.png')

    plt.clf()

    #K=15
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)
    sn.heatmap(confusion)
    plt.savefig('/Users/matthewli/Documents/Spotify Project/Confusion_Matrix.png')
    print(classification_report(y_test, y_pred))

    #Predicting
    knn.fit(x, y)
    predictdata = get_playlist_tracks('1256293535', '4ttw16DvJncoO4aY4c7wmU', '')
    predictdata = pd.DataFrame(predictdata, columns=['Artist', 'Name', 'ID', 'Playlist', 'Energy', 'Liveness', 'Tempo',
                                                     'Speechiness', 'Acousticness', 'Instrumentalness', 'Danceability',
                                                     'Key', 'Duration', 'Loudness', 'Valence', 'Mode'])
    if len(predictdata) > 0:
        predictvalues = predictdata.drop(predictdata.columns[0:4], axis=1)
        for x in ['Duration', 'Tempo', 'Key', 'Loudness', 'Instrumentalness', 'Mode', 'Liveness']:
            del predictvalues[x]
        print(predictvalues)
        print(knn.predict(predictvalues.values.tolist()))
        predictdata['Predict'] = knn.predict(predictvalues.values.tolist())
    predictdata.to_csv('/Users/matthewli/Documents/Spotify Project/Predict.csv')

#Create Playlist
def create_playlist():
    data = pd.read_csv('/Users/matthewli/Documents/Spotify Project/Spotify_PCA.csv')
    for x in data.loc[data['Clusters'] == 0]['ID']:
        print(x)
        spotify.user_playlist_add_tracks('1256293535', '00W1ZozOzYbVMVYvmE8CL5', [x])
    for x in data.loc[data['Clusters'] == 1]['ID']:
        print(x)
        spotify.user_playlist_add_tracks('1256293535', '2nqkTLR8cmOj2FL3JJtOz0', [x])
    for x in data.loc[data['Clusters'] == 2]['ID']:
        print(x)
        spotify.user_playlist_add_tracks('1256293535', '3vFarZjPRZsgtXUVE9ELty', [x])

#Add to Playlist
def add_to_playlist():
    predictdata = pd.read_csv('/Users/matthewli/Documents/Spotify Project/Predict.csv')
    if len(predictdata) > 0:
        for x in predictdata.loc[predictdata['Predict'] == 0]['ID']:
            print(x)
            spotify.user_playlist_add_tracks('1256293535', '00W1ZozOzYbVMVYvmE8CL5', [x])
            spotify.user_playlist_remove_all_occurrences_of_tracks('1256293535', '4ttw16DvJncoO4aY4c7wmU', [x])
        for x in predictdata.loc[predictdata['Predict'] == 1]['ID']:
            print(x)
            spotify.user_playlist_add_tracks('1256293535', '2nqkTLR8cmOj2FL3JJtOz0', [x])
            spotify.user_playlist_remove_all_occurrences_of_tracks('1256293535', '4ttw16DvJncoO4aY4c7wmU', [x])
        for x in predictdata.loc[predictdata['Predict'] == 2]['ID']:
            print(x)
            spotify.user_playlist_add_tracks('1256293535', '3vFarZjPRZsgtXUVE9ELty', [x])
            spotify.user_playlist_remove_all_occurrences_of_tracks('1256293535', '4ttw16DvJncoO4aY4c7wmU', [x])

get_my_songs()
pcadata = read_data('/Users/matthewli/Documents/Spotify Project/Spotify.csv')
pca(pcadata)
knn(pcadata)
add_to_playlist()


