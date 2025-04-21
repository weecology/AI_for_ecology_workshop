import os
import random
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from opensoundscape import Audio, Spectrogram
from opensoundscape.metrics import predict_multi_target_labels
from opensoundscape.ml.shallow_classifier import MLPClassifier, fit_classifier_on_embeddings
import bioacoustics_model_zoo as bmz

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="Audio object is shorter than requested duration",
    module="opensoundscape.audio"
)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
pd.options.mode.chained_assignment = None


def analyze_species_distribution(train_df):
    """
    Analyze the species distribution in the dataset.
    """
    print("\n=== Species Distribution Analysis ===")
    species_counts = train_df['primary_label'].value_counts()

    plt.figure(figsize=(12, 6))
    species_counts.head(20).plot(kind='bar')
    plt.title('Top 20 Most Common Species')
    plt.xlabel('Species')
    plt.ylabel('Record Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"Total unique species: {len(species_counts)}")
    print(f"Most common species: {species_counts.index[0]} ({species_counts.iloc[0]} records)")
    print(f"Least common species: {species_counts.index[-1]} ({species_counts.iloc[-1]} records)")


def prepare_data():
    """
    Load and prepare the dataset for training and testing.
    """
    print("Loading data...")
    train_df = pd.read_csv("collaboration_challenge/birdclef-2025/train.csv")
    taxonomy_df = pd.read_csv("collaboration_challenge/birdclef-2025/taxonomy.csv")
    sample_submission = pd.read_csv("collaboration_challenge/birdclef-2025/sample_submission.csv")

    train_df["full_path"] = train_df.apply(
        lambda row: f"collaboration_challenge/birdclef-2025/train_audio/{row['filename']}", axis=1
    )
    test_df = train_df.groupby('primary_label').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
    train_df = train_df[~train_df.index.isin(test_df.index)]
    test_df = test_df[test_df.primary_label.isin(train_df.primary_label.unique())]

    labels = train_df.primary_label.unique()

    return train_df, test_df, taxonomy_df, labels


def train_model(train_df, test_df, labels):
    """
    Train the model using the prepared data.
    """
    formatted_train_df = pd.concat([train_df] * 12, ignore_index=True)
    formatted_train_df["start_time"] = (formatted_train_df.index % 12) * 5
    formatted_train_df["end_time"] = formatted_train_df["start_time"] + 5
    formatted_train_df.rename(columns={"full_path": "file"}, inplace=True)

    formatted_test_df = pd.concat([test_df] * 12, ignore_index=True)
    formatted_test_df["start_time"] = (formatted_test_df.index % 12) * 5
    formatted_test_df["end_time"] = formatted_test_df["start_time"] + 5
    formatted_test_df.rename(columns={"full_path": "file"}, inplace=True)

    formatted_train_df = pd.get_dummies(
        formatted_train_df[["file", "start_time", "end_time", "primary_label"]]
        .set_index(["file", "start_time", "end_time"])["primary_label"]
    ) * 1
    formatted_test_df = pd.get_dummies(
        formatted_test_df[["file", "start_time", "end_time", "primary_label"]]
        .set_index(["file", "start_time", "end_time"])["primary_label"]
    ) * 1

    m = bmz.BirdSetEfficientNetB1()
    m.change_classes(labels)

    clf = MLPClassifier(
        input_size=1280, output_size=formatted_train_df.shape[1], hidden_layer_sizes=()
    )

    emb_train, label_train, emb_val, label_val = fit_classifier_on_embeddings(
        embedding_model=m,
        classifier_model=clf,
        train_df=formatted_train_df,
        validation_df=formatted_test_df,
        steps=1000,
        embedding_batch_size=12,
        embedding_num_workers=5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    preds = clf(emb_val.to(torch.device("cuda"))).cpu().detach().numpy()
    print(f"Average precision score: {average_precision_score(label_val.cpu(), preds, average=None)}")
    print(f"Area under ROC: {roc_auc_score(label_val.cpu(), preds, average=None)}")
    return m, clf


def predict(test_soundscapes, clf, model, labels):
    """
    Predict species in test soundscapes.
    """
    predictions = []
    for soundscape in test_soundscapes:
        # Load and embed
        embedding = model.embed(soundscape)
        embedding_values = torch.tensor(embedding.values, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        scores = clf(embedding_values)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.cpu().detach().numpy()
        
        # Softmax
        prediction = pd.DataFrame(scores, columns=labels, index=embedding.index)
        
        predictions.append(prediction)
    
    predictions = pd.concat(predictions)
    predictions.to_csv('submission.csv', index=False)
    
    return predictions


# Main execution
train_df, test_df, taxonomy_df, labels = prepare_data()
analyze_species_distribution(train_df)
m, classifier = train_model(train_df, test_df, labels)

# Test the model on a train soundscape
train_soundscape_path = 'collaboration_challenge/birdclef-2025/train_soundscapes/'
train_soundscapes = glob.glob(os.path.join(train_soundscape_path, '*.ogg'))
predictions = predict(train_soundscapes, classifier, m, labels)
print(predictions.head())

# get predicted labels for each train_soundscape
predicted_labels = predictions.idxmax(axis=1).reset_index().value_counts()

# For running only on Kaggle
# test_soundscape_path = 'collaboration_challenge/birdclef-2025/test_soundscapes/'
# test_soundscapes = glob.glob(os.path.join(test_soundscape_path, '*.ogg'))
# predictions = predict(test_soundscapes, classifier, m, labels)
# print(predictions.head())


# Clean labels - activity weak labels
# Semi-supervision using train_soundscapes using prebuilt embeddings
# Train augmentation
# Test augmentation
# Model ensembling