"""
=============================================================================
MLP (Multilayer Perceptron) for Power System Fault Classification
=============================================================================
Architecture  : 12 → 512 → 256 → 128 → 64 → 5  ("Funnel" design)
Fault Classes : Healthy, LG, LL, LLG, LLLG
Dataset Source: MATLAB/Simulink wavelet-extracted features (dataset.xlsx)
Target Acc.   : ≥ 95.78 %
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH   = os.path.join(os.path.dirname(__file__), "..", "Santosh", "dataset.xlsx")
INPUT_DIM      = 12          # 12 features (sequence components + wavelet coefficients)
NUM_CLASSES    = 5           # Healthy, LG, LL, LLG, LLLG
FAULT_LABELS   = ["Healthy", "LG", "LL", "LLG", "LLLG"]
TEST_SIZE      = 0.20        # 80 / 20 split
BATCH_SIZE     = 32
MAX_EPOCHS     = 200
PATIENCE       = 7           # Early-stopping patience (5-10 range)
LEARNING_RATE  = 1e-3
RANDOM_STATE   = 42
TARGET_ACC     = 95.78       # benchmark accuracy (%)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path: str):
    """
    Load the dataset and return scaled features + one-hot encoded labels.

    Expected format
    ───────────────
    The Excel/CSV file should have:
      • Feature columns  – 12 numeric columns (sequence components & wavelet
        coefficients extracted from MATLAB/Simulink).
      • Label column     – A column named 'Fault_Type' (or the *last* column)
        containing the fault class strings.

    Returns
    -------
    X_train, X_test : np.ndarray   – Scaled feature arrays
    y_train, y_test : np.ndarray   – One-hot encoded label arrays
    label_encoder   : LabelEncoder – Fitted encoder (for inverse transform)
    scaler          : StandardScaler
    """
    # ---------- read file ----------
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    print(f"[INFO] Dataset loaded  →  shape {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    # ---------- separate features & labels ----------
    # Try to find label column by common names; fall back to last column
    label_col = None
    for candidate in ["Fault_Type", "fault_type", "FaultType", "label", "Label", "class", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]
        print(f"[WARN] No standard label column found – using last column: '{label_col}'")

    X = df.drop(columns=[label_col]).values.astype(np.float32)
    y_raw = df[label_col].values

    # ---------- encode labels ----------
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    y_onehot = keras.utils.to_categorical(y_int, num_classes=len(le.classes_))
    print(f"[INFO] Classes detected: {list(le.classes_)}")
    print(f"[INFO] Feature dimension: {X.shape[1]}")

    # ---------- train / test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_int,
    )

    # ---------- feature scaling ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[INFO] Train samples : {X_train.shape[0]}")
    print(f"[INFO] Test  samples : {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test, le, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MODEL DEFINITION  —  "Funnel" MLP Architecture
# ──────────────────────────────────────────────────────────────────────────────
def build_mlp(input_dim: int, num_classes: int) -> keras.Model:
    """
    Construct the deep MLP with a progressive compression ("funnel") design.

    Layer Stack
    ───────────
    Input  →  Dense(512, ReLU)  →  Dense(256, ReLU)  →
              Dense(128, ReLU)  →  Dense(64,  ReLU)  →  Dense(num_classes, Softmax)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,), name="input_features"),

        # Hidden Layer 1 – capture initial non-linear transient relationships
        layers.Dense(512, activation="relu", name="hidden_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.3, name="dropout_1"),

        # Hidden Layer 2
        layers.Dense(256, activation="relu", name="hidden_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.25, name="dropout_2"),

        # Hidden Layer 3
        layers.Dense(128, activation="relu", name="hidden_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.Dropout(0.2, name="dropout_3"),

        # Hidden Layer 4 – final feature abstraction
        layers.Dense(64, activation="relu", name="hidden_4"),
        layers.BatchNormalization(name="bn_4"),
        layers.Dropout(0.15, name="dropout_4"),

        # Output Layer – probability distribution over fault classes
        layers.Dense(num_classes, activation="softmax", name="output_fault_class"),
    ], name="Fault_Classification_MLP")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3.  COMPILATION & OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────
def compile_model(model: keras.Model) -> keras.Model:
    """
    Compile with:
      • Loss      : Categorical Cross-Entropy  (penalises severe misclassification)
      • Optimizer  : Adam  (adaptive learning rate)
      • Metrics    : Accuracy
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING PROTOCOL
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train with early stopping on val_loss and a learning-rate reducer.

    Returns
    -------
    history : keras.callbacks.History
    """
    cb_list = [
        # Early Stopping – halt if val_loss stagnates for `PATIENCE` epochs
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on plateau for finer convergence
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION & ACCURACY VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate on the test set:
      • Print classification report
      • Generate & plot confusion matrix
      • Check against the 95.78 % accuracy benchmark
    """
    # ---- predictions ----
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ---- accuracy check ----
    acc = accuracy_score(y_true, y_pred) * 100
    print("=" * 60)
    print(f"  TEST ACCURACY : {acc:.2f} %")
    print(f"  TARGET        : {TARGET_ACC} %")
    if acc >= TARGET_ACC:
        print("  ✅  BENCHMARK PASSED")
    else:
        print(f"  ⚠️  Below target by {TARGET_ACC - acc:.2f} pp – consider more data or tuning.")
    print("=" * 60, "\n")

    # ---- classification report ----
    class_names = list(label_encoder.classes_)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---- confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Fault")
    plt.ylabel("Actual Fault")
    plt.title(f"Confusion Matrix  —  Accuracy {acc:.2f} %")
    plt.tight_layout()

    # Save figure next to the script
    fig_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[INFO] Confusion matrix saved → {fig_path}")
    plt.show()

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING HISTORY PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_history(history):
    """Plot accuracy & loss curves for training vs. validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Accuracy ----
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"],  label="Val Accuracy")
    axes[0].axhline(y=TARGET_ACC / 100, color="r", linestyle="--", label=f"Target {TARGET_ACC}%")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Loss ----
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss  (Categorical Cross-Entropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), "training_history.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[INFO] Training history saved → {fig_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  MLP Fault Classifier  –  Sanepa Distribution Network")
    print("=" * 60 + "\n")

    # ── Step 1: Load & preprocess ──
    X_train, X_test, y_train, y_test, le, scaler = load_and_preprocess(DATASET_PATH)

    actual_input_dim = X_train.shape[1]
    actual_num_classes = y_train.shape[1]

    if actual_input_dim != INPUT_DIM:
        print(f"[WARN] Expected {INPUT_DIM} features but dataset has {actual_input_dim}. "
              f"Adapting model input dimension automatically.\n")

    if actual_num_classes != NUM_CLASSES:
        print(f"[WARN] Expected {NUM_CLASSES} classes but dataset has {actual_num_classes}. "
              f"Adapting output layer automatically.\n")

    # ── Step 2: Build model ──
    model = build_mlp(input_dim=actual_input_dim, num_classes=actual_num_classes)
    model = compile_model(model)
    model.summary()

    # ── Step 3: Train ──
    history = train_model(model, X_train, y_train, X_test, y_test)

    # ── Step 4: Evaluate ──
    acc = evaluate_model(model, X_test, y_test, le)

    # ── Step 5: Visualise training curves ──
    plot_training_history(history)

    # ── Step 6: Save model ──
    model_path = os.path.join(os.path.dirname(__file__), "fault_mlp_model.keras")
    model.save(model_path)
    print(f"\n[INFO] Trained model saved → {model_path}")
    print("[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
