from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np
import pandas as pd
import csv



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('StandardScaler.pkl', 'rb'))



@app.route("/upload_file", methods=["GET"])
def hello():
    return render_template("upload_file.html")

@app.route("/")
@app.route("/final", methods=["GET"])
def hellou():
    return render_template("final.html")



@app.route("/upload_file", methods=["POST"])
def upload():

    filen = request.files["file"]

    if filen.filename == "":
        return render_template("home.html")
    else:
        fpath = "./files/" + filen.filename
        filen.save(fpath)

        y, sr = librosa.load(fpath, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, n_mfcc=13, sr=sr)
        header = 'chroma_stft spectral_centroid rolloff zero_crossing_rate'
        for i in range(1, 14):
            header += f' mfcc{i}'
        header = header.split()
        to_append = f'{np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(rolloff)} {np.mean(zcr)}'
        file = open('testing.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        file = open('testing.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

        d1 = pd.read_csv('testing.csv')
        d1 = np.asarray(d1)
        d1 = d1.reshape(1, -1)
        d1 = sc.transform(d1)
        genre = {0: 'BLUES', 1: 'CLASSICAL', 2: 'COUNTRY', 3: 'JAZZ', 4: 'METAL', 5: 'POP'}
        pred_genre = model.predict(d1)

        return render_template("after.html", prediction=genre[pred_genre[0]])




