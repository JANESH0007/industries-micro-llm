"""
Udyam NIC Code Predictor — Training Script
===========================================
Ministry of Industries | TensorFlow

USE CASE:
  A small business owner types what they do in plain English.
  The model predicts the correct NIC 2008 sub-class code
  for their Udyam / MSME registration.

EXAMPLE:
  Input : "i make soap at home"
  Output: NIC 20231 — Manufacture of soap (all forms)
          Division : Manufacturing
          Register : Udyam Portal (udyamregistration.gov.in)

ARCHITECTURE: Dual-output Transformer
  → Output 1: NIC code classification
  → Output 2: Division classification (Agriculture / Manufacturing / Trade etc.)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json, os

print("TensorFlow version:", tf.__version__)

# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_PATH = "data/nic_2008_synthesized.csv"

df_synth = pd.read_csv(DATA_PATH)

# Also load original industries.csv for better training examples
df_orig = pd.read_csv("data/industries.csv")
df_orig = df_orig.rename(
    columns={"text": "Description", "nic_code": "Sub Class", "division": "Division"}
)
df_orig["Sub Class"] = df_orig["Sub Class"].astype(str)
df_orig = df_orig[["Description", "Sub Class", "Division"]]

# Filter synthesized data to remove "Services related to"
mask = ~df_synth["Description"].str.contains(
    "Services related to", case=False, regex=True
)
df_synth_filtered = df_synth[mask].copy()

# Combine both datasets
df = pd.concat([df_orig, df_synth_filtered], ignore_index=True)
df = df.drop_duplicates(subset=["Description", "Sub Class"])

print(f"\nLoaded {len(df)} samples (combined)")
print(f"Unique Sub Class codes : {df['Sub Class'].nunique()}")
print(f"Unique divisions : {df['Division'].nunique()}")

texts = df["Description"].astype(str).values
nic_codes = df["Sub Class"].astype(str).values
divisions = df["Division"].astype(str).values

print(f"\nDEBUG: nic_codes unique = {len(set(nic_codes))}")
print(f"DEBUG: divisions unique = {len(set(divisions))}")
print(f"DEBUG: sample divisions = {list(set(divisions))[:5]}")

# =====================================================
# 2. ENCODE LABELS
# =====================================================

nic_encoder = LabelEncoder()
div_encoder = LabelEncoder()

y_nic = nic_encoder.fit_transform(nic_codes)
y_div = div_encoder.fit_transform(divisions)

num_nic = len(nic_encoder.classes_)
num_div = len(div_encoder.classes_)

print(f"DEBUG: num_nic = {num_nic}, num_div = {num_div}")

print(f"\nNIC classes  : {num_nic}")
print(f"Division classes: {num_div} → {list(div_encoder.classes_)}")

# Save label maps
label_map = {
    "nic": {str(i): c for i, c in enumerate(nic_encoder.classes_)},
    "division": {str(i): c for i, c in enumerate(div_encoder.classes_)},
}

# Also save nic_code → Description mapping for lookup
nic_label_map = {}
for _, row in df[["Sub Class", "Description"]].drop_duplicates().iterrows():
    nic_label_map[str(row["Sub Class"])] = row["Description"]

label_map["nic_labels"] = nic_label_map

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

print("Label maps saved → label_map.json")

# =====================================================
# 3. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, yn_train, yn_test, yd_train, yd_test = train_test_split(
    texts, y_nic, y_div, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# =====================================================
# 4. TEXT VECTORIZATION
# =====================================================

MAX_VOCAB = 12000
SEQ_LEN = 40
EMBED_DIM = 128

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
    name="text_vectorizer",
)
vectorizer.adapt(X_train)

X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)

# Save vocabulary
vocab_config = {
    "vocabulary": vectorizer.get_vocabulary(),
    "max_vocab": MAX_VOCAB,
    "seq_len": SEQ_LEN,
}
with open("vectorizer_vocab.json", "w") as f:
    json.dump(vocab_config, f)

print(f"Vocabulary size: {len(vocab_config['vocabulary'])}")

# =====================================================
# 5. TRANSFORMER BLOCK (serializable)
# =====================================================


@tf.keras.utils.register_keras_serializable(package="UdyamNIC")
class TransformerBlock(tf.keras.layers.Layer):
    """Self-attention + FFN encoder block"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


# =====================================================
# 6. BUILD DUAL-OUTPUT MICRO LLM
# =====================================================


def build_model(vocab_size, embed_dim, seq_len, num_heads, ff_dim, num_nic, num_div):

    inputs = tf.keras.Input(shape=(seq_len,), name="token_ids")

    # Embeddings
    tok_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, name="token_embedding"
    )(inputs)

    pos_emb = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=embed_dim, name="position_embedding"
    )(tf.range(seq_len))

    x = tok_emb + pos_emb
    x = tf.keras.layers.Dropout(0.1)(x)

    # Transformer layers
    x = TransformerBlock(embed_dim, num_heads, ff_dim, name="transformer_1")(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, name="transformer_2")(x)

    # Pooling → Shared representation
    x = tf.keras.layers.GlobalAveragePooling1D(name="pooling")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    shared = tf.keras.layers.Dense(128, activation="relu", name="shared")(x)

    # Output 1: NIC sub-class code
    nic_out = tf.keras.layers.Dense(num_nic, activation="softmax", name="nic_out")(
        shared
    )

    # Output 2: Division (broad sector)
    div_out = tf.keras.layers.Dense(num_div, activation="softmax", name="div_out")(
        shared
    )

    model = tf.keras.Model(
        inputs=inputs, outputs=[nic_out, div_out], name="Udyam_NIC_MicroLLM"
    )
    return model


model = build_model(
    vocab_size=MAX_VOCAB,
    embed_dim=EMBED_DIM,
    seq_len=SEQ_LEN,
    num_heads=4,
    ff_dim=256,
    num_nic=num_nic,
    num_div=num_div,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "nic_out": "sparse_categorical_crossentropy",
        "div_out": "sparse_categorical_crossentropy",
    },
    loss_weights={"nic_out": 0.8, "div_out": 0.2},
    metrics={
        "nic_out": "accuracy",
        "div_out": "accuracy",
    },
)

# model.summary()
print(f"\n✅ Total parameters: {model.count_params():,}  (minimum 1000 ✔)")

# =====================================================
# 7. TRAIN
# =====================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_nic_out_accuracy",
        patience=5,
        verbose=1,
        mode="max",
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_nic_out_accuracy",
        verbose=1,
        mode="max",
    ),
]

print("\n--- Training ---")
history = model.fit(
    X_train_vec,
    {"nic_out": yn_train, "div_out": yd_train},
    validation_data=(X_test_vec, {"nic_out": yn_test, "div_out": yd_test}),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
)

# =====================================================
# 8. EVALUATE
# =====================================================

results = model.evaluate(
    X_test_vec, {"nic_out": yn_test, "div_out": yd_test}, verbose=0
)

print(f"\nNIC Code Accuracy  : {results[3] * 100:.1f}%")
print(f"Division Accuracy  : {results[4] * 100:.1f}%")

# =====================================================
# 9. SAVE
# =====================================================

try:
    model.save("udyam_nic_model.keras")
    print("Model save SUCCESS")
except Exception as e:
    print(f"Model save FAILED: {e}")

import os

print("\nModel saved  : udyam_nic_model.keras", os.path.exists("udyam_nic_model.keras"))
print("Vocab saved  : vectorizer_vocab.json", os.path.exists("vectorizer_vocab.json"))
print("Labels saved : label_map.json", os.path.exists("label_map.json"))
