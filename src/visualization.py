import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd

def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform") -> None:
    """Plot the waveform of an audio signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_melspectrogram(mel_spec: np.ndarray, sr: int, title: str = "Mel Spectrogram") -> None:
    """Plot the mel spectrogram of an audio signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], 
                         labels: List[str], title: str = "Confusion Matrix") -> None:
    """Plot confusion matrix for model predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_training_history(history: dict, title: str = "Training History") -> None:
    """Plot training and validation metrics over epochs."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_prediction_confidence(predictions: np.ndarray, labels: List[str], 
                             title: str = "Prediction Confidence") -> None:
    """Plot confidence scores for each class prediction."""
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, predictions)
    plt.yticks(y_pos, labels)
    plt.xlabel('Confidence Score')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_augmentation_comparison(original: np.ndarray, augmented: np.ndarray, 
                               sr: int, title: str = "Augmentation Comparison") -> None:
    """Compare original and augmented audio signals."""
    plt.figure(figsize=(12, 6))
    
    # Plot waveforms
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original, sr=sr)
    plt.title('Original Audio')
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(augmented, sr=sr)
    plt.title('Augmented Audio')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(features: np.ndarray, labels: List[str], 
                            title: str = "Feature Distribution") -> None:
    """Plot distribution of features using PCA."""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(set(labels)):
        mask = np.array(labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=label, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attention_weights(attention_weights: torch.Tensor, 
                         title: str = "Attention Weights") -> None:
    """Plot attention weights from the model."""
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Attention Heads')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels: List[str], title: str = "Class Distribution") -> None:
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=(12, 6))
    class_counts = pd.Series(labels).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_audio_duration_distribution(durations: List[float], 
                                   title: str = "Audio Duration Distribution") -> None:
    """Plot the distribution of audio durations in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=30)
    plt.title(title)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_matrix(features: np.ndarray, 
                                  feature_names: List[str],
                                  title: str = "Feature Correlation Matrix") -> None:
    """Plot correlation matrix between different features."""
    plt.figure(figsize=(12, 10))
    corr_matrix = np.corrcoef(features.T)
    sns.heatmap(corr_matrix, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_tsne_visualization(features: np.ndarray, labels: List[str],
                          title: str = "t-SNE Visualization") -> None:
    """Plot t-SNE visualization of features."""
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 8))
    for label in set(labels):
        mask = np.array(labels) == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   label=label, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_architecture(model: torch.nn.Module, 
                          input_shape: Tuple[int, ...],
                          title: str = "Model Architecture") -> None:
    """Plot model architecture summary."""
    from torchviz import make_dot
    import torch
    
    # Create a dummy input
    dummy_input = torch.randn(input_shape)
    
    # Generate the graph
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    
    # Save and display the graph
    dot.render("model_architecture", format="png")
    plt.figure(figsize=(15, 10))
    plt.imshow(plt.imread("model_architecture.png"))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_gradient_flow(model: torch.nn.Module, 
                      title: str = "Gradient Flow") -> None:
    """Plot the flow of gradients through the model layers."""
    plt.figure(figsize=(12, 6))
    
    # Get gradients
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append((name, param.grad.norm().item()))
    
    # Plot
    names = [g[0] for g in gradients]
    values = [g[1] for g in gradients]
    
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(names)), names, rotation=45)
    plt.title(title)
    plt.xlabel('Gradient Norm')
    plt.tight_layout()
    plt.show()

def plot_learning_rate_schedule(optimizer: torch.optim.Optimizer,
                              num_epochs: int,
                              title: str = "Learning Rate Schedule") -> None:
    """Plot the learning rate schedule over training epochs."""
    plt.figure(figsize=(10, 6))
    
    # Get learning rates
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
    
    plt.plot(range(num_epochs), lrs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_analysis(y_true: List[str], y_pred: List[str],
                       probabilities: np.ndarray,
                       labels: List[str],
                       title: str = "Error Analysis") -> None:
    """Plot detailed error analysis of model predictions."""
    plt.figure(figsize=(15, 10))
    
    # Get misclassified samples
    misclassified = np.array(y_true) != np.array(y_pred)
    
    # Plot confusion matrix for misclassified samples
    plt.subplot(2, 1, 1)
    cm = confusion_matrix(np.array(y_true)[misclassified], 
                         np.array(y_pred)[misclassified],
                         labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Misclassified Samples)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot confidence distribution for correct vs incorrect predictions
    plt.subplot(2, 1, 2)
    correct_probs = probabilities[~misclassified]
    incorrect_probs = probabilities[misclassified]
    
    sns.kdeplot(correct_probs, label='Correct Predictions')
    sns.kdeplot(incorrect_probs, label='Incorrect Predictions')
    plt.title('Confidence Distribution')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Density')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 