from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import io
import pickle  

import librosa
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import joblib
mel_scaler = joblib.load('./models/mel_scaler.pkl')
max_duration = joblib.load('./models/max_duration.pkl')

app = FastAPI()

# templates 폴더 생성
os.makedirs('templates', exist_ok=True)

# Jinja2 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "transcript": None})

   

@app.post("/", response_class=HTMLResponse)
def upload_audio(request: Request, audio: UploadFile = File(...)):
    # 실제 변환 없이 업로드 파일 이름만 결과로 표시
    
    file_name = audio.filename

    print(file_name + " 업로드 완료")
    print("--------------------------------")

    audio_bytes = audio.file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    print("오디오 읽는 중")
    y, sr = librosa.load(audio_buffer, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    ## 위에서 사용한 mel_scaler 재사용함
    mel_db = mel_scaler.transform(mel_db.reshape(-1, 1)).reshape(mel_db.shape)
    mel_db = np.expand_dims(mel_db, axis=-1) # (128, T) → (128, T, 1)


    # 길이 부족하면 반복해서 붙이고 잘라내기
    T = mel_db.shape[1]
    if T < max_duration:
        repeat = int(np.ceil(max_duration / T))
        mel_db = np.concatenate([mel_db] * repeat, axis=1)[:, :max_duration, :]
    else:
        mel_db = mel_db[:, :max_duration, :]


    mel_db = np.expand_dims(mel_db, axis=0) # (1, 128, 1876, 1)
    print("오디오 읽기 완료")
    print("--------------------------------")

    print("분석 모델 불러오는 중")
    model_loudness = load_model("./models/deep_learning/loudness_model.keras")
    loudness_scaler = MinMaxScaler() # 추론 시점에 새로 다시 선언
    loudness_scaler.data_min_ = np.load("./models/deep_learning/loudness_scaler_min_.npy")
    loudness_scaler.scale_ = np.load("./models/deep_learning/loudness_scaler_scale_.npy")
    loudness_scaler.min_ = 0
    loudness_scaler.data_max_ = loudness_scaler.data_min_ + 1 / loudness_scaler.scale_

    model_valence = load_model("./models/deep_learning/valence_model.keras")
    valence_scaler = MinMaxScaler() # 추론 시점에 새로 다시 선언
    valence_scaler.data_min_ = np.load("./models/deep_learning/valence_scaler_min_.npy")
    valence_scaler.scale_ = np.load("./models/deep_learning/valence_scaler_scale_.npy")
    valence_scaler.min_ = 0
    valence_scaler.data_max_ = valence_scaler.data_min_ + 1 / valence_scaler.scale_

    model_tempo = load_model("./models/deep_learning/tempo_model.keras")
    tempo_scaler = MinMaxScaler() # 추론 시점에 새로 다시 선언
    tempo_scaler.data_min_ = np.load("./models/deep_learning/tempo_scaler_min_.npy")
    tempo_scaler.scale_ = np.load("./models/deep_learning/tempo_scaler_scale_.npy")
    tempo_scaler.min_ = 0
    tempo_scaler.data_max_ = tempo_scaler.data_min_ + 1 / tempo_scaler.scale_
    print("모델 불러오기 완료")
    print("--------------------------------")

    print("loudness, valence, tempo 예측 중")

    pred_loudness_scaled = model_loudness.predict(mel_db)
    pred_loudness = loudness_scaler.inverse_transform(pred_loudness_scaled)

    pred_valence_scaled = model_valence.predict(mel_db)
    pred_valence = valence_scaler.inverse_transform(pred_valence_scaled)

    pred_tempo_scaled = model_tempo.predict(mel_db)
    pred_tempo = tempo_scaler.inverse_transform(pred_tempo_scaled)

    print("loudness, valence, tempo 예측 완료")
    

    deep_learning_result = {
        'loudness': float(f"{pred_loudness[0][0]:.2f}"),
        'valence': float(f"{pred_valence[0][0]:.2f}"),
        'tempo': float(f"{pred_tempo[0][0]:.2f}"),
    }

    print(f"예측 결과 - {deep_learning_result}")
    print("--------------------------------")

    ##################################딥러닝####################################

    print("loudness, valence, tempo를 통한 popularity 예측 중")

    with open("./models/machine_learning/spotifyPred.dump","rb") as popularity_predictor:
        mlCore = pickle.load(popularity_predictor)

        inLoudness = deep_learning_result['loudness']
        inValence = deep_learning_result['valence']
        inTempo = deep_learning_result['tempo']

        futureDf = pd.DataFrame( [[inLoudness,inValence,inTempo]] )

        PredictDt = mlCore.predict(futureDf)

    print("loudness, valence, tempo를 통한 popularity 예측 완료료")
    print("--------------------------------")
    transcript = f'Spotify의 경향에 따르면, 사용자가 업로드한 {file_name}의 인기도는 {round(PredictDt[0], 2)}로 추정됩니다.' 
    return templates.TemplateResponse("index.html", {"request": request, "transcript": transcript})


if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False) 