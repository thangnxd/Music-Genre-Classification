import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import librosa

# =========================
# 1. Định nghĩa model CNN-BiLSTM cho demo phụ
# =========================
class MusicCNNLSTM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.lstm = nn.LSTM(
            128 * 16, 256, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        _, (hn, _) = self.lstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(x)


# =========================
# 2. Path cấu trúc project
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TASK_B_DIR = os.path.join(RESULTS_DIR, "task_b")
TASK_C_DIR = os.path.join(RESULTS_DIR, "task_c")
FINAL_DIR = os.path.join(RESULTS_DIR, "final")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TASK_B_SUMMARY = os.path.join(TASK_B_DIR, "final_summary.csv")
TASK_C_METRICS = os.path.join(TASK_C_DIR, "dl_metrics.csv")
DL_PREDICTIONS = os.path.join(TASK_C_DIR, "dl_predictions.pkl")
LEARNING_CURVE = os.path.join(TASK_C_DIR, "learning_curve.png")

MODEL_COMPARISON = os.path.join(FINAL_DIR, "model_comparison.csv")
CONFUSION_MATRIX = os.path.join(FINAL_DIR, "cnn_lstm_confusion_matrix.png")
ERROR_ANALYSIS = os.path.join(FINAL_DIR, "genre_error_analysis.csv")
CLASSIFICATION_REPORT = os.path.join(FINAL_DIR, "cnn_lstm_classification_report.csv")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_best.pt")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "ds1_knn.pkl")
KNN_SCALER_PATH = os.path.join(MODEL_DIR, "ds1_scaler.pkl")
KNN_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "ds1_label_encoder.pkl")

# SVM có thể được lưu với các tên khác nhau tùy notebook train.
# Dashboard sẽ tự tìm file tồn tại đầu tiên trong danh sách này.
SVM_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "ds1_svm.pkl"),
    os.path.join(MODEL_DIR, "svm.pkl"),
    os.path.join(MODEL_DIR, "ds1_svm_model.pkl"),
    os.path.join(MODEL_DIR, "svm_model.pkl"),
]

RF_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "ds1_random_forest.pkl"),
    os.path.join(MODEL_DIR, "random_forest.pkl"),
    os.path.join(MODEL_DIR, "ds1_rf.pkl"),
    os.path.join(MODEL_DIR, "rf.pkl"),
]

XGBOOST_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "ds1_xgboost.pkl"),
    os.path.join(MODEL_DIR, "xgboost.pkl"),
    os.path.join(MODEL_DIR, "ds1_xgb.pkl"),
    os.path.join(MODEL_DIR, "xgb.pkl"),
]

MODELS_METADATA_PATH = os.path.join(MODEL_DIR, "models_metadata.json")

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Feature order phổ biến của GTZAN features_3_sec.csv
DEFAULT_FEATURE_NAMES = [
    "chroma_stft_mean", "chroma_stft_var",
    "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var",
    "perceptr_mean", "perceptr_var",
    "tempo",
]
for i in range(1, 21):
    DEFAULT_FEATURE_NAMES += [f"mfcc{i}_mean", f"mfcc{i}_var"]


# =========================
# 3. Cấu hình giao diện
# =========================
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classification Dashboard")
st.caption("Task D - Đánh giá mô hình, trực quan hóa và demo dự đoán thể loại nhạc")


def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def show_file_warning(path):
    st.warning(f"Không tìm thấy file: `{path}`")


@st.cache_resource
def load_knn_assets():
    if not os.path.exists(KNN_MODEL_PATH):
        raise FileNotFoundError(KNN_MODEL_PATH)
    if not os.path.exists(KNN_SCALER_PATH):
        raise FileNotFoundError(KNN_SCALER_PATH)
    if not os.path.exists(KNN_LABEL_ENCODER_PATH):
        raise FileNotFoundError(KNN_LABEL_ENCODER_PATH)

    knn = joblib.load(KNN_MODEL_PATH)
    scaler = joblib.load(KNN_SCALER_PATH)
    label_encoder = joblib.load(KNN_LABEL_ENCODER_PATH)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = DEFAULT_FEATURE_NAMES

    return knn, scaler, label_encoder, feature_names



def find_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_svm_assets():
    svm_path = find_existing_path(SVM_MODEL_CANDIDATES)
    if svm_path is None:
        raise FileNotFoundError(
            "Không tìm thấy SVM model. Hãy đặt file vào một trong các tên: "
            + ", ".join(os.path.basename(p) for p in SVM_MODEL_CANDIDATES)
        )
    if not os.path.exists(KNN_SCALER_PATH):
        raise FileNotFoundError(KNN_SCALER_PATH)
    if not os.path.exists(KNN_LABEL_ENCODER_PATH):
        raise FileNotFoundError(KNN_LABEL_ENCODER_PATH)

    svm = joblib.load(svm_path)
    scaler = joblib.load(KNN_SCALER_PATH)
    label_encoder = joblib.load(KNN_LABEL_ENCODER_PATH)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = DEFAULT_FEATURE_NAMES

    return svm, scaler, label_encoder, feature_names, svm_path




@st.cache_resource
def load_rf_assets():
    rf_path = find_existing_path(RF_MODEL_CANDIDATES)
    if rf_path is None:
        raise FileNotFoundError(
            "Không tìm thấy Random Forest model. Hãy đặt file vào một trong các tên: "
            + ", ".join(os.path.basename(p) for p in RF_MODEL_CANDIDATES)
        )
    if not os.path.exists(KNN_SCALER_PATH):
        raise FileNotFoundError(KNN_SCALER_PATH)
    if not os.path.exists(KNN_LABEL_ENCODER_PATH):
        raise FileNotFoundError(KNN_LABEL_ENCODER_PATH)

    rf = joblib.load(rf_path)
    scaler = joblib.load(KNN_SCALER_PATH)
    label_encoder = joblib.load(KNN_LABEL_ENCODER_PATH)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = DEFAULT_FEATURE_NAMES

    return rf, scaler, label_encoder, feature_names, rf_path


@st.cache_resource
def load_xgboost_assets():
    xgb_path = find_existing_path(XGBOOST_MODEL_CANDIDATES)
    if xgb_path is None:
        raise FileNotFoundError(
            "Không tìm thấy XGBoost model. Hãy đặt file vào một trong các tên: "
            + ", ".join(os.path.basename(p) for p in XGBOOST_MODEL_CANDIDATES)
        )
    if not os.path.exists(KNN_SCALER_PATH):
        raise FileNotFoundError(KNN_SCALER_PATH)
    if not os.path.exists(KNN_LABEL_ENCODER_PATH):
        raise FileNotFoundError(KNN_LABEL_ENCODER_PATH)

    xgb = joblib.load(xgb_path)
    scaler = joblib.load(KNN_SCALER_PATH)
    label_encoder = joblib.load(KNN_LABEL_ENCODER_PATH)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = DEFAULT_FEATURE_NAMES

    return xgb, scaler, label_encoder, feature_names, xgb_path


def render_traditional_ml_demo(
    tab,
    model_name,
    load_assets_func,
    upload_key,
    button_key,
    description,
    extra_warning=None,
):
    """Render demo cho các model ML truyền thống dùng chung feature GTZAN 3s."""
    with tab:
        st.header(f"Demo thêm - Dự đoán bằng {model_name}")
        st.write(description)
        st.info(
            "Demo cắt toàn bộ bài nhạc thành các đoạn 3 giây, "
            "extract feature cho từng đoạn, dự đoán từng đoạn, rồi lấy trung bình xác suất/điểm tin cậy."
        )

        if extra_warning:
            st.warning(extra_warning)

        uploaded_file = st.file_uploader(
            f"Tải lên file nhạc để dự đoán bằng {model_name}",
            type=["wav", "mp3", "ogg"],
            key=upload_key
        )

        if uploaded_file is None:
            st.info(f"Hãy upload một file nhạc để bắt đầu dự đoán bằng {model_name}.")
            return

        st.audio(uploaded_file)

        if st.button(f"Dự đoán bằng {model_name}", key=button_key):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name

                model, scaler, label_encoder, feature_names, model_path = load_assets_func()

                X_features, y_audio, sr = extract_features_for_knn(
                    audio_path,
                    feature_names,
                    segment_seconds=3.0,
                    max_segments=None
                )
                X_scaled = scaler.transform(X_features)

                avg_probs = predict_average_probabilities(model, X_scaled, label_encoder)

                pred_idx = int(np.argmax(avg_probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(avg_probs[pred_idx])

                c1, c2, c3 = st.columns(3)
                c1.metric("Thể loại dự đoán", pred_label.upper())
                c2.metric("Độ tin cậy", f"{confidence * 100:.2f}%")
                c3.metric("Số đoạn 3s phân tích", len(X_features))

                st.caption(f"Model đang dùng: `{model_path}`")

                if not hasattr(model, "predict_proba"):
                    st.warning(
                        f"{model_name} này không có predict_proba. Dashboard đang quy đổi decision_function/vote "
                        "sang phân phối tương đối, nên độ tin cậy chỉ mang tính tham khảo."
                    )

                if confidence < 0.5:
                    st.warning(
                        "Dự đoán có độ tin cậy thấp. File nhạc có thể nằm ngoài phân phối dữ liệu GTZAN "
                        "hoặc có nhiều thể loại pha trộn."
                    )

                prob_df = pd.DataFrame({
                    "Genre": label_encoder.classes_,
                    "Probability": avg_probs
                }).sort_values("Probability", ascending=False)
                prob_df["Probability (%)"] = prob_df["Probability"] * 100

                st.subheader("Top dự đoán")
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.set_index("Genre")["Probability"])

                st.subheader("Mel Spectrogram tham khảo")
                mel = librosa.feature.melspectrogram(
                    y=y_audio,
                    sr=sr,
                    n_mels=128,
                    n_fft=2048,
                    hop_length=512
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(mel_db, aspect="auto", origin="lower")
                ax.set_title("Mel Spectrogram")
                ax.set_xlabel("Time")
                ax.set_ylabel("Mel bands")
                fig.colorbar(img, ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(
                    f"Demo {model_name} bị lỗi. Hãy kiểm tra file model trong thư mục models "
                    "và các file ds1_scaler.pkl, ds1_label_encoder.pkl."
                )
                st.exception(e)


@st.cache_resource
def load_cnn_model():
    model = MusicCNNLSTM(num_classes=10)
    state_dict = torch.load(CNN_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================
# 4. Feature extraction cho KNN
# =========================
def _mean_var(x):
    return float(np.mean(x)), float(np.var(x))


def extract_gtzan_features_from_segment(y, sr):
    """Trích xuất feature giống nhóm feature_3_sec của GTZAN."""
    features = {}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_stft_mean"], features["chroma_stft_var"] = _mean_var(chroma)

    rms = librosa.feature.rms(y=y)
    features["rms_mean"], features["rms_var"] = _mean_var(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"], features["spectral_centroid_var"] = _mean_var(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"], features["spectral_bandwidth_var"] = _mean_var(bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"], features["rolloff_var"] = _mean_var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_rate_mean"], features["zero_crossing_rate_var"] = _mean_var(zcr)

    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y)
    features["harmony_mean"], features["harmony_var"] = _mean_var(harmony)
    features["perceptr_mean"], features["perceptr_var"] = _mean_var(perceptr)

    try:
        tempo = librosa.feature.tempo(y=y, sr=sr)
        features["tempo"] = float(np.ravel(tempo)[0])
    except Exception:
        features["tempo"] = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f"mfcc{i}_mean"] = float(np.mean(mfcc[i - 1]))
        features[f"mfcc{i}_var"] = float(np.var(mfcc[i - 1]))

    return features


def extract_features_for_knn(audio_path, feature_names, sr=22050, segment_seconds=3.0, max_segments=None):
    """
    Cắt audio thành các segment 3s rồi extract feature cho từng segment.

    Bản cũ dùng max_segments=10, tức là chỉ lấy 10 đoạn x 3s = 30 giây đầu.
    Bản này mặc định max_segments=None để lấy toàn bộ bài hát:
    60s -> 20 segment, 180s -> 60 segment.
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    segment_length = int(segment_seconds * sr)

    rows = []
    total_segments = len(y) // segment_length

    if max_segments is not None:
        total_segments = min(max_segments, total_segments)

    for i in range(total_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]

        if len(segment) < segment_length:
            continue

        feat_dict = extract_gtzan_features_from_segment(segment, sr)
        row = [feat_dict.get(name, 0.0) for name in feature_names]
        rows.append(row)

    if not rows:
        # fallback: nếu file quá ngắn thì pad về 3 giây
        if len(y) < segment_length:
            y = np.pad(y, (0, segment_length - len(y)), mode="constant")
        feat_dict = extract_gtzan_features_from_segment(y[:segment_length], sr)
        rows.append([feat_dict.get(name, 0.0) for name in feature_names])

    return np.array(rows), y, sr


def predict_average_probabilities(model, X_scaled, label_encoder):
    """
    Trả về xác suất trung bình trên các segment.
    - Nếu model có predict_proba: dùng trực tiếp.
    - Nếu SVM không bật probability=True: dùng decision_function và softmax.
    - Nếu không có cả hai: dùng vote theo nhãn predict.
    """
    if hasattr(model, "predict_proba"):
        segment_probs = model.predict_proba(X_scaled)
        return segment_probs.mean(axis=0)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_scaled)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        segment_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return segment_probs.mean(axis=0)

    preds = model.predict(X_scaled)
    avg_probs = np.zeros(len(label_encoder.classes_), dtype=float)
    for pred in preds:
        idx = int(pred)
        avg_probs[idx] += 1
    avg_probs = avg_probs / max(1, len(preds))
    return avg_probs


# =========================
# 5. Load dữ liệu kết quả
# =========================
task_b = safe_read_csv(TASK_B_SUMMARY)
task_c = safe_read_csv(TASK_C_METRICS)
model_comparison = safe_read_csv(MODEL_COMPARISON)

# =========================
# 6. Tabs
# =========================
tabs = st.tabs([
    "Tổng quan",
    "Kết quả ML truyền thống",
    "Kết quả Deep Learning",
    "So sánh mô hình",
    "Demo KNN",
    "Demo SVM",
    "Demo Random Forest",
    "Demo XGBoost",
    "Demo CNN-BiLSTM"
])


with tabs[0]:
    st.header("Tổng quan dự án")

    c1, c2, c3 = st.columns(3)
    c1.metric("Bài toán", "Music Genre Classification")
    c2.metric("Dataset chính", "GTZAN")
    c3.metric("Số lớp", "10 genres")

    st.markdown("""
    Dự án xây dựng hệ thống phân loại thể loại nhạc dựa trên dữ liệu âm thanh.

    **Các thành phần chính:**
    - **Task B:** Huấn luyện các mô hình Machine Learning truyền thống trên đặc trưng âm thanh dạng bảng.
    - **Task C:** Huấn luyện mô hình Deep Learning CNN2D + BiLSTM trên Mel Spectrogram.
    - **Task D:** Tổng hợp kết quả, so sánh mô hình, trực quan hóa và xây dựng dashboard demo.
    """)

    st.subheader("10 thể loại trong GTZAN")
    st.write(", ".join([g.capitalize() for g in GENRES]))

    st.subheader("Cấu trúc project")
    st.code("""
Music-Genre-Classification/
├── app/
│   └── dashboard.py
├── models/
│   ├── ds1_knn.pkl
│   ├── ds1_scaler.pkl
│   ├── ds1_label_encoder.pkl
│   └── cnn_lstm_best.pt
├── results/
│   ├── task_b/
│   ├── task_c/
│   └── final/
└── notebooks/
    └── task-d.ipynb
    """)


with tabs[1]:
    st.header("Kết quả Task B - Machine Learning truyền thống")

    if task_b is None:
        show_file_warning(TASK_B_SUMMARY)
    else:
        st.subheader("Bảng kết quả")
        st.dataframe(task_b, use_container_width=True)

        acc_col = "DS1 Test" if "DS1 Test" in task_b.columns else None
        if acc_col and "Model" in task_b.columns:
            st.subheader("Accuracy trên DS1 Test")
            chart_df = task_b[["Model", acc_col]].rename(columns={acc_col: "Accuracy"})
            st.bar_chart(chart_df.set_index("Model"))

            best = chart_df.sort_values("Accuracy", ascending=False).iloc[0]
            st.success(f"Mô hình ML tốt nhất trên DS1: {best['Model']} - Accuracy = {best['Accuracy']:.4f}")
        else:
            st.info("Chưa tìm thấy cột DS1 Test để vẽ biểu đồ.")
    
    if task_b is not None and {"Model", "DS1 Test", "DS2 Test"}.issubset(task_b.columns):
        st.subheader("So sánh Accuracy giữa DS1 và DS2")

        compare_df = task_b[["Model", "DS1 Test", "DS2 Test"]].set_index("Model")
        st.bar_chart(compare_df)

        st.caption(
            "Biểu đồ này cho thấy mô hình nào không chỉ cao trên DS1 mà còn giữ được hiệu quả trên DS2."
        )


with tabs[2]:
    st.header("Kết quả Task C - Deep Learning")

    if task_c is None:
        show_file_warning(TASK_C_METRICS)
    else:
        st.subheader("Chỉ số đánh giá CNN2D + BiLSTM")
        st.dataframe(task_c, use_container_width=True)

        row = task_c.iloc[0]
        c1, c2, c3 = st.columns(3)
        if "accuracy" in task_c.columns:
            c1.metric("Accuracy", f"{row['accuracy']:.4f}")
        if "f1_macro" in task_c.columns:
            c2.metric("F1 Macro", f"{row['f1_macro']:.4f}")
        if "f1_weighted" in task_c.columns:
            c3.metric("F1 Weighted", f"{row['f1_weighted']:.4f}")
        if "notes" in task_c.columns:
            st.info(row["notes"])

    st.subheader("Learning Curve")
    if os.path.exists(LEARNING_CURVE):
        st.image(LEARNING_CURVE, use_container_width=True)
    else:
        show_file_warning(LEARNING_CURVE)

    st.subheader("Confusion Matrix - CNN2D + BiLSTM")
    st.caption("Ma trận này cho biết mô hình phân loại đúng/sai giữa các thể loại như thế nào.")
    if os.path.exists(CONFUSION_MATRIX):
        st.image(CONFUSION_MATRIX, use_container_width=True)
    else:
        show_file_warning(CONFUSION_MATRIX)

    report_df = safe_read_csv(CLASSIFICATION_REPORT)
    if report_df is not None:
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

    error_analysis = safe_read_csv(ERROR_ANALYSIS)
    if error_analysis is not None:
        st.subheader("Phân tích lỗi theo từng thể loại")
        st.dataframe(error_analysis, use_container_width=True)
        if "Genre" in error_analysis.columns and "Class Accuracy" in error_analysis.columns:
            st.bar_chart(error_analysis.set_index("Genre")["Class Accuracy"])


with tabs[3]:
    st.header("So sánh mô hình")

    st.info(
        "Task B và Task C dùng hai pipeline khác nhau. "
        "Task B dùng đặc trưng âm thanh dạng bảng từ features_3_sec.csv với các mô hình ML truyền thống. "
        "Task C dùng Mel Spectrogram từ audio thô với CNN2D + BiLSTM. "
        "Vì vậy bảng so sánh có ý nghĩa tham khảo, đồng thời cho thấy mô hình nào hiệu quả hơn trong từng pipeline."
    )

    if model_comparison is None:
        show_file_warning(MODEL_COMPARISON)
    else:
        st.subheader("Bảng so sánh tổng hợp")
        st.dataframe(model_comparison, use_container_width=True)

        if "Model" in model_comparison.columns and "Accuracy" in model_comparison.columns:
            st.subheader("Accuracy Comparison")
            st.bar_chart(model_comparison.set_index("Model")["Accuracy"])

            best = model_comparison.sort_values("Accuracy", ascending=False).iloc[0]
            st.success(f"🏆 Mô hình tốt nhất: {best['Model']} - Accuracy = {best['Accuracy']:.4f}")


with tabs[4]:
    st.header("Demo chính - Dự đoán bằng KNN")
    st.write("Mô hình KNN được chọn làm demo chính vì đạt accuracy cao nhất trên DS1 GTZAN.")
    st.info("Demo hiện cắt toàn bộ bài nhạc thành các đoạn 3 giây, rồi lấy trung bình xác suất trên tất cả các đoạn. Bản cũ chỉ lấy tối đa 10 đoạn đầu, tương đương 30 giây đầu.")

    uploaded_file = st.file_uploader(
        "Tải lên file nhạc để dự đoán bằng KNN",
        type=["wav", "mp3", "ogg"],
        key="knn_upload"
    )

    if uploaded_file is None:
        st.info("Hãy upload một file nhạc để bắt đầu dự đoán.")
    else:
        st.audio(uploaded_file)

        if st.button("Dự đoán bằng KNN", key="knn_predict"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name

                knn, scaler, label_encoder, feature_names = load_knn_assets()

                X_features, y_audio, sr = extract_features_for_knn(
                    audio_path,
                    feature_names,
                    segment_seconds=3.0,
                    max_segments=None
                )
                X_scaled = scaler.transform(X_features)

                avg_probs = predict_average_probabilities(knn, X_scaled, label_encoder)

                pred_idx = int(np.argmax(avg_probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(avg_probs[pred_idx])

                c1, c2, c3 = st.columns(3)
                c1.metric("Thể loại dự đoán", pred_label.upper())
                c2.metric("Độ tin cậy", f"{confidence * 100:.2f}%")
                c3.metric("Số đoạn 3s phân tích", len(X_features))

                if confidence < 0.5:
                    st.warning(
                        "Dự đoán có độ tin cậy thấp. File nhạc có thể nằm ngoài phân phối dữ liệu GTZAN "
                        "hoặc có nhiều thể loại pha trộn."
                    )

                prob_df = pd.DataFrame({
                    "Genre": label_encoder.classes_,
                    "Probability": avg_probs
                }).sort_values("Probability", ascending=False)
                prob_df["Probability (%)"] = prob_df["Probability"] * 100

                st.subheader("Top dự đoán")
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.set_index("Genre")["Probability"])

                st.subheader("Mel Spectrogram tham khảo")
                mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(mel_db, aspect="auto", origin="lower")
                ax.set_title("Mel Spectrogram")
                ax.set_xlabel("Time")
                ax.set_ylabel("Mel bands")
                fig.colorbar(img, ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error("Demo KNN bị lỗi. Hãy kiểm tra đủ file ds1_knn.pkl, ds1_scaler.pkl và ds1_label_encoder.pkl trong thư mục models.")
                st.exception(e)


with tabs[5]:
    st.header("Demo thêm - Dự đoán bằng SVM")
    st.write(
        "SVM dùng cùng bộ đặc trưng GTZAN 3 giây như KNN. "
        "Dashboard sẽ cắt toàn bộ bài nhạc thành các đoạn 3 giây, dự đoán từng đoạn, rồi lấy trung bình xác suất/điểm tin cậy."
    )

    uploaded_file = st.file_uploader(
        "Tải lên file nhạc để dự đoán bằng SVM",
        type=["wav", "mp3", "ogg"],
        key="svm_upload"
    )

    if uploaded_file is None:
        st.info("Hãy upload một file nhạc để bắt đầu dự đoán bằng SVM.")
    else:
        st.audio(uploaded_file)

        if st.button("Dự đoán bằng SVM", key="svm_predict"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name

                svm, scaler, label_encoder, feature_names, svm_path = load_svm_assets()

                X_features, y_audio, sr = extract_features_for_knn(
                    audio_path,
                    feature_names,
                    segment_seconds=3.0,
                    max_segments=None
                )
                X_scaled = scaler.transform(X_features)

                avg_probs = predict_average_probabilities(svm, X_scaled, label_encoder)

                pred_idx = int(np.argmax(avg_probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(avg_probs[pred_idx])

                c1, c2, c3 = st.columns(3)
                c1.metric("Thể loại dự đoán", pred_label.upper())
                c2.metric("Độ tin cậy tương đối", f"{confidence * 100:.2f}%")
                c3.metric("Số đoạn 3s phân tích", len(X_features))

                st.caption(f"Model đang dùng: `{svm_path}`")

                if not hasattr(svm, "predict_proba"):
                    st.warning(
                        "SVM này không có predict_proba. Dashboard đang quy đổi decision_function sang phân phối tương đối bằng softmax, "
                        "nên độ tin cậy chỉ mang tính tham khảo."
                    )

                prob_df = pd.DataFrame({
                    "Genre": label_encoder.classes_,
                    "Probability": avg_probs
                }).sort_values("Probability", ascending=False)
                prob_df["Probability (%)"] = prob_df["Probability"] * 100

                st.subheader("Top dự đoán")
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.set_index("Genre")["Probability"])

                st.subheader("Mel Spectrogram tham khảo")
                mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(mel_db, aspect="auto", origin="lower")
                ax.set_title("Mel Spectrogram")
                ax.set_xlabel("Time")
                ax.set_ylabel("Mel bands")
                fig.colorbar(img, ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(
                    "Demo SVM bị lỗi. Hãy kiểm tra file SVM trong thư mục models, ví dụ: "
                    "ds1_svm.pkl hoặc svm.pkl. SVM dùng chung ds1_scaler.pkl và ds1_label_encoder.pkl."
                )
                st.exception(e)


render_traditional_ml_demo(
    tab=tabs[6],
    model_name="Random Forest",
    load_assets_func=load_rf_assets,
    upload_key="rf_upload",
    button_key="rf_predict",
    description=(
        "Random Forest dùng cùng bộ đặc trưng GTZAN 3 giây như KNN/SVM. "
        "Model này thường ổn định và dễ giải thích hơn các mô hình phức tạp."
    )
)


render_traditional_ml_demo(
    tab=tabs[7],
    model_name="XGBoost",
    load_assets_func=load_xgboost_assets,
    upload_key="xgb_upload",
    button_key="xgb_predict",
    description=(
        "XGBoost dùng cùng bộ đặc trưng GTZAN 3 giây như các model ML truyền thống khác. "
        "Đây là model boosting mạnh, thường cho kết quả tốt trên dữ liệu đặc trưng dạng bảng."
    ),
    extra_warning=(
        "Nếu gặp lỗi `No module named xgboost`, hãy cài thêm `xgboost` vào môi trường hoặc requirements.txt."
    )
)


with tabs[8]:
    st.header("Demo phụ - CNN2D + BiLSTM")
    st.write("Phần này minh họa pipeline Deep Learning xử lý trực tiếp Mel Spectrogram từ audio.")

    uploaded_file = st.file_uploader(
        "Tải lên file nhạc để dự đoán bằng CNN-BiLSTM",
        type=["wav", "mp3", "ogg"],
        key="cnn_upload"
    )

    if uploaded_file is None:
        st.info("Hãy upload một file nhạc để thử demo CNN-BiLSTM.")
    else:
        st.audio(uploaded_file)

        if st.button("Dự đoán bằng CNN-BiLSTM", key="cnn_predict"):
            try:
                import librosa

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name

                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)

                st.subheader("Mel Spectrogram")
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(mel_db, aspect="auto", origin="lower")
                ax.set_title("Mel Spectrogram")
                ax.set_xlabel("Time")
                ax.set_ylabel("Mel bands")
                fig.colorbar(img, ax=ax)
                st.pyplot(fig)

                if not os.path.exists(CNN_MODEL_PATH):
                    st.warning("Không tìm thấy cnn_lstm_best.pt nên chưa thể predict bằng CNN-BiLSTM.")
                    st.stop()
                if not os.path.exists(DL_PREDICTIONS):
                    st.warning("Không tìm thấy dl_predictions.pkl nên chưa lấy được danh sách genre.")
                    st.stop()

                with open(DL_PREDICTIONS, "rb") as f:
                    pred_data = pickle.load(f)
                genres = pred_data.get("genres", GENRES)

                model = load_cnn_model()

                target_frames = 130
                if mel_db.shape[1] < target_frames:
                    pad_width = target_frames - mel_db.shape[1]
                    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
                else:
                    mel_db = mel_db[:, :target_frames]

                mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
                x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_genre = genres[pred_idx]
                confidence = float(probs[pred_idx])

                st.success(f"Dự đoán CNN-BiLSTM: **{pred_genre.upper()}**")
                st.metric("Độ tin cậy", f"{confidence * 100:.2f}%")

                if confidence < 0.5:
                    st.warning("Mô hình CNN-BiLSTM chưa thật sự tự tin với file này.")

                prob_df = pd.DataFrame({"Genre": genres, "Probability": probs}).sort_values("Probability", ascending=False)
                prob_df["Probability (%)"] = prob_df["Probability"] * 100
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.set_index("Genre")["Probability"])

            except Exception as e:
                st.error("Demo CNN-BiLSTM bị lỗi. Dashboard vẫn dùng được.")
                st.exception(e)
