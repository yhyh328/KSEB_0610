# 🎵 Spotify 음원 인기도 반응 웹 애플리케이션

범 프로젝트는 Spotify 메타데이터(특히 `loudness` 등 오디오 특성)들을 기본으로 음원의 **인기도(popularity)** 와의 상관관계를 분석하고, FastAPI 웹 인터페이스를 통해 예측 결과를 제공합니다.

---

## 🔍 주요 기능

- **오디오 특성 분석**
  - `loudness`, `valence`, `tempo` 등 해당 특성 추출
  - 그룹별(`groupby`) 상관관계 분석을 통해 의미 있는 피처 생성
- **인기도 예측 모델**
  - `scikit-learn` 기반 상관계수 분석
  - `TensorFlow` 모델을 이용한 회규/분류
- **웹 인터페이스**
  - FastAPI 서버
  - MP3/WAV 파일 업로드 → 분석 결과 리포트 & 시각화

---

## 📆 사용 기술 스택

| 범주    | 기술                       |
| ----- | ------------------------ |
| 언어    | Python (3.8 \~ 3.10 권장)  |
| 프레임워크 | FastAPI                  |
| 머신링   | Scikit-learn, TensorFlow |
| 시각화   | Matplotlib, Seaborn      |
| 기타    | Pandas, Librosa, Joblib  |

---

## 🚀 설치 및 실행

### 1. Python 버전

- 권장: **3.8 ≤ Python ≤ 3.10**
- ‼️ Python 3.11 이상은 TensorFlow 호환 문제 발생 가능성 있음

### 2. 가상환경 (선택)

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate.bat      # Windows
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 파일 구조 확인

```text
project_root/
├── main.py
├── requirements.txt
├── README.md
├── colabs/              # 주피터 노트북 파일들
│   ├── deep_learning_models.ipynb
│   └── machine_learning_model.ipynb
├── models/             # 학습된 모델 파일들
│   ├── max_duration.pkl
│   ├── mel_scaler.pkl
│   ├── deep_learning/  # 딥러닝 모델 관련 파일
│   │   ├── loudness_model.keras
│   │   ├── tempo_model.keras
│   │   ├── valence_model.keras
│   │   └── [scaler files]
│   └── machine_learning/  # 머신러닝 모델 관련 파일
│       └── spotifyPred.dump
└── templates/          # 웹 인터페이스 템플릿
    └── index.html
```

### 5. 서버 시작

```bash
python main.py
```

### 6. 브라우저 접속

```
http://localhost:8000/
```

---

## ⚠️ 주의사항

- 지원 파일 형식: **.mp3**, **.wav**
- 그 외 형식 업로드 시 오류 발생
- `static/` 폴더를 만들지 않으면 FastAPI 정적 파일 로딩에서 오류 발생

---

## 📖 활용 기술

- **Groupby + Apply**
  - `pandas.DataFrame.groupby()` + `apply()` 을 통해 그룹 단위 분석
- **상관관계 분석**
  - `scikit-learn` + `numpy` 를 이용해 수치형 피처가 관련 관계에 있는지 및 가장 중요한 피처 선별
- **TensorFlow 모델**
  - 예측 모델 (.keras) 및 전처리 모델 (.joblib)
