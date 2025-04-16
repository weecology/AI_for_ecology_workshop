from opensoundscape.ml.cnn import load_model
from opensoundscape import Audio, Spectrogram
from opensoundscape.metrics import predict_multi_target_labels
from opensoundscape.ml.shallow_classifier import MLPClassifier, fit_classifier_on_embeddings
import pandas as pd
import matplotlib.pyplot as plt
import bioacoustics_model_zoo as bmz
import torch
import random
import numpy as np
import os
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
pd.options.mode.chained_assignment = None  # Suppress pandas warnings

def analizar_distribucion_especies(train_df):
    """
    Analiza la distribución de especies en el conjunto de datos.
    Args:
        train_df (DataFrame): DataFrame de entrenamiento
    """
    print("\n=== Análisis de Distribución de Especies ===")
    
    # Análisis de distribución de especies
    species_counts = train_df['primary_label'].value_counts()
    
    # Visualización de las 20 especies más comunes
    plt.figure(figsize=(12, 6))
    species_counts.head(20).plot(kind='bar')
    plt.title('Top 20 Especies más Comunes')
    plt.xlabel('Especie')
    plt.ylabel('Cantidad de Registros')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print(f"\nTotal de especies únicas: {len(species_counts)}")
    print(f"Especie más común: {species_counts.index[0]} ({species_counts.iloc[0]} registros)")
    print(f"Especie menos común: {species_counts.index[-1]} ({species_counts.iloc[-1]} registros)")

    # Visualización de las 20 especies más comunes
    plt.figure(figsize=(12, 6))
    species_counts.plot(kind='bar')
    plt.xlabel('Especie')
    plt.ylabel('Cantidad de Registros')

def cargar_datos():
    """
    Carga los archivos principales del conjunto de datos.
    Returns:
        tuple: DataFrames de entrenamiento, taxonomía y muestra
    """
    print("Cargando datos...")
    
    # Carga de archivos principales
    train_df = pd.read_csv("collaboration_challenge/birdclef-2025/train.csv")
    taxonomy_df = pd.read_csv("collaboration_challenge/birdclef-2025/taxonomy.csv")
    sample_submission = pd.read_csv("collaboration_challenge/birdclef-2025/sample_submission.csv")
    
    # Carga de metadatos de ubicación
    with open("collaboration_challenge/birdclef-2025/recording_location.txt", "r") as f:
        recording_location = f.read()
    
    print(f"Datos de entrenamiento: {train_df.shape}")
    print(f"Datos taxonómicos: {taxonomy_df.shape}")
    print(f"Archivo de muestra: {sample_submission.shape}")
    
    return train_df, taxonomy_df, sample_submission, recording_location

train_df, taxonomy_df, sample_submission, recording_location = cargar_datos()

analizar_distribucion_especies(train_df)
plt.show()

# Opensoundscape and BirdCLEF
bmz.utils.list_models()

# Load the model
m = bmz.BirdSetEfficientNetB1()

Crested_Bobwhite = train_df[train_df.common_name == "Crested Bobwhite"]
Crested_Bobwhite.head()

# Load the audio
file_path = "collaboration_challenge/birdclef-2025/train_audio/crebob1/XC148253.ogg"
audio = Audio.from_file(file_path)
fft_spectrum, frequencies = audio.spectrum()

# Plot
plt.plot(frequencies, fft_spectrum)
plt.ylabel('Fast Fourier Transform (V**2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.show()

# Low pass filter
clean_audio = audio.reduce_noise().highpass(1000, order=8).lowpass(5000, order=8).normalize()
fft_spectrum, frequencies = clean_audio.spectrum()

# Plot
plt.plot(frequencies, fft_spectrum)
plt.ylabel('Fast Fourier Transform (V**2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.show()
clean_audio.show_widget()

spectrogram_object = Spectrogram.from_audio(clean_audio)
spectrogram_object.plot()
plt.show()

clean_audio.save("clean_audio.wav")
scores = m.predict("clean_audio.wav", activation_layer="sigmoid") 

predicted_labels = predict_multi_target_labels(scores, threshold=0.5)
detection_counts = predicted_labels.sum(0)
detections = detection_counts[detection_counts > 0]

print(detections)

# Matching taxonomy
taxonomy_df.loc[taxonomy_df.primary_label.isin(detections.index)]

m.freeze_feature_extractor()

# Create train/test split with 1 random sample per class for test
# Full path to the images
train_df["full_path"] = train_df.apply(
    lambda row: f"collaboration_challenge/birdclef-2025/train_audio/{row['filename']}", axis=1
)
test_df = train_df.groupby('primary_label').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
train_df = train_df[~train_df.index.isin(test_df.index)]

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

labels = train_df.primary_label.unique()
m.change_classes(labels)

# Create a mini dataset for testing, one train sample for each species
train_df = train_df.groupby('primary_label').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

formatted_train_df = pd.get_dummies(train_df[["full_path", "primary_label"]].set_index("full_path")["primary_label"]) * 1
formatted_test_df = pd.get_dummies(test_df[["full_path", "primary_label"]].set_index("full_path")["primary_label"]) * 1

formatted_train_df

embeddings = m.embed(formatted_train_df, batch_size=128, num_workers=2)

# if os.path.exists("collaboration_challenge/benweinstein/embeddings.csv"):
#     embeddings = pd.read_csv("collaboration_challenge/benweinstein/embeddings.csv")
# else:
#     embeddings = m.embed(formatted_train_df, batch_size=128, num_workers=2)
#     embeddings.to_csv("collaboration_challenge/benweinstein/embeddings.csv", index=False)

utclf = MLPClassifier(
    input_size=1280, output_size=formatted_train_df.shape[1], hidden_layer_sizes=()
)

emb_train, label_train, emb_val, label_val = fit_classifier_on_embeddings(
    embedding_model=m,
    classifier_model=clf,
    train_df=formatted_train_df,
    validation_df=formatted_test_df,
    steps=3,
    embedding_batch_size=128,
    embedding_num_workers=5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

preds = clf(emb_val.to(torch.device("cuda"))).detach().numpy()
# evaluate with threshold agnostic metrics: MAP and ROC AUC
print(
    f"average precision score: {average_precision_score(label_val,preds,average=None)}"
)
print(f"area under ROC: {roc_auc_score(label_val,preds,average=None)}")