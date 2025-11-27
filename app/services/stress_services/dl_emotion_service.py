# -*- coding: utf-8 -*-
"""
ONNX 기반 감정 예측 서비스
TensorFlow 제거 버전
"""

import io
import os
import numpy as np
import librosa
import onnxruntime as ort
from typing import Tuple


def _resolve_under_app(relative_path: str) -> str:
    services_dir = os.path.dirname(os.path.abspath(__file__))  # .../stress_services
    app_dir = os.path.dirname(os.path.dirname(services_dir))   # .../app
    return os.path.normpath(os.path.join(app_dir, relative_path))


class EmotionDLService:
    def __init__(self) -> None:
        # ---- 모델 경로 설정 ----
        env_model = os.getenv("EMOTION_MODEL_PATH", "").strip()
        env_label = os.getenv("EMOTION_LABEL_PATH", "").strip()

        default_model_rel = os.path.join("models", "stresscare_models", "emotion_model.onnx")
        default_label_rel = os.path.join("models", "stresscare_models", "emotion_labels_all.npy")

        self.model_path = (
            _resolve_under_app(env_model) if env_model else _resolve_under_app(default_model_rel)
        )
        self.label_path = (
            _resolve_under_app(env_label) if env_label else _resolve_under_app(default_label_rel)
        )

        self.unknown_threshold = float(os.getenv("EMOTION_UNKNOWN_THRESHOLD", "0.50"))

        # ---- 모델 로드 ----
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX 모델이 없습니다: {self.model_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"라벨 파일이 없습니다: {self.label_path}")

        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.labels = np.load(self.label_path, allow_pickle=True).tolist()

        # ONNX 모델 입력 이름 가져오기
        self.input_name = self.session.get_inputs()[0].name

        print("---- EmotionDLService (ONNX) Loaded ----")
        print(f"MODEL: {self.model_path}")
        print(f"LABEL: {self.label_path}")
        print("-----------------------------------------")

    # 음성 → MelSpectrogram 변환
    @staticmethod
    def _extract_melspec_from_audio(
        y: np.ndarray,
        sr: int,
        n_mels: int = 128,
        duration: float = 3.0,
    ) -> np.ndarray:

        target_len = int(sr * duration)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mn, mx = mel_db.min(), mel_db.max()
        if mx - mn != 0:
            mel_db = (mel_db - mn) / (mx - mn)

        return mel_db[..., np.newaxis].astype(np.float32)

    # ONNX 예측
    def predict_emotion_from_bytes(self, audio_bytes: bytes, sr: int = 22050) -> Tuple[str, float]:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

        X = self._extract_melspec_from_audio(y, sr)
        X = np.expand_dims(X, axis=0)  # 모델 입력: (1, 128, T, 1)

        outputs = self.session.run(None, {self.input_name: X})
        probs = outputs[0][0]  # 첫 번째 출력 가져오기

        idx = int(np.argmax(probs))
        prob = float(probs[idx])

        if prob < self.unknown_threshold:
            return "unknown", prob

        return self.labels[idx], prob

    def predict_emotion_from_file(self, path: str, sr: int = 22050) -> Tuple[str, float]:
        with open(path, "rb") as f:
            return self.predict_emotion_from_bytes(f.read(), sr=sr)
