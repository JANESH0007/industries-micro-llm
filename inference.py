"""
Udyam NIC Code Predictor — Inference
=====================================
Ministry of Industries | TensorFlow

Run after training: python train_model.py
Usage: python inference.py

A common person or MSME officer types a business description
in plain English and gets the correct NIC 2008 code, Udyam
registration guidance, and compliance requirements.
"""

import tensorflow as tf
import numpy as np
import json
import sys

# =====================================================
# 1. RE-DEFINE TRANSFORMER (must match training)
# =====================================================

@tf.keras.utils.register_keras_serializable(package="UdyamNIC")
class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.ff_dim       = ff_dim
        self.dropout_rate = dropout

        self.att   = tf.keras.layers.MultiHeadAttention(
                         num_heads=num_heads,
                         key_dim=embed_dim // num_heads)
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn(x)
        ffn  = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout_rate,
        })
        return config

# =====================================================
# 2. UDYAM COMPLIANCE LOOKUP
#    Division → what registration/compliance is needed
# =====================================================

UDYAM_GUIDANCE = {
    "Agriculture": {
        "portal":      "Kisan Suvidha / PM-Kisan Portal",
        "registration": "PM-KISAN (pmkisan.gov.in)",
        "schemes":     ["PM-KISAN", "PMFBY Crop Insurance", "Kisan Credit Card"],
        "compliance":  "APEDA registration if exporting",
    },
    "Manufacturing": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["PMEGP", "Mudra Yojana", "CGTMSE", "PLI Scheme"],
        "compliance":  "GST, Factory License, BIS Certification",
    },
    "Construction": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["PMEGP", "Mudra Yojana"],
        "compliance":  "RERA Registration, Labour License, GST",
    },
    "Trade": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Mudra Yojana", "Stand Up India"],
        "compliance":  "GST Registration, Shop License",
    },
    "Transport": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Mudra Yojana", "PM Gati Shakti"],
        "compliance":  "RTO Permit, National Transit Pass",
    },
    "Hospitality": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Mudra Yojana", "PMEGP"],
        "compliance":  "FSSAI Food License, GST, Shop License",
    },
    "IT Services": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Startup India", "Digital India"],
        "compliance":  "GST Registration, Shop License",
    },
    "Finance": {
        "portal":      "RBI / SEBI / IRDAI",
        "registration": "RBI Registration (for lending)",
        "schemes":     ["Stand Up India", "CGTMSE"],
        "compliance":  "RBI NBFC, SEBI Registration, IRDAI",
    },
    "Real Estate": {
        "portal":      "RERA Portal",
        "registration": "RERA (state-specific portal)",
        "schemes":     ["PMAY", "NHB Schemes"],
        "compliance":  "RERA Registration, GST",
    },
    "Professional": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Startup India", "Mudra Yojana"],
        "compliance":  "Professional Tax, GST",
    },
    "Education": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["PMKVY", "NSDC Skill Development"],
        "compliance":  "AICTE/UGC Approval, NSDC Accreditation",
    },
    "Healthcare": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Ayushman Bharat", "PLI Devices"],
        "compliance":  "NABH, CDSCO, Drug License",
    },
    "Services": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Mudra Yojana", "PMEGP"],
        "compliance":  "GST, Shop & Establishment License",
    },
    "Energy": {
        "portal":      "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":     ["Solar Subsidy", "Green Hydrogen Mission"],
        "compliance":  "CEA Guidelines, Grid Approval",
    },
    "Mining": {
        "portal":      "IBM (ibm.gov.in)",
        "registration": "IBM Clearance",
        "schemes":     ["Mineral Exploration Trust"],
        "compliance":  "Mining Lease, IBM Clearance, PCB Consent",
    },
}

# =====================================================
# 3. LOAD MODEL + VOCAB + LABELS
# =====================================================

print("\nLoading Udyam NIC Predictor Engine...")

try:
    model = tf.keras.models.load_model("udyam_nic_model.keras")
    
    with open("label_map.json") as f:
        label_map = json.load(f)

    nic_map    = label_map["nic"]
    div_map    = label_map["division"]
    nic_labels = label_map["nic_labels"]

    with open("vectorizer_vocab.json") as f:
        vocab_config = json.load(f)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_config["max_vocab"],
        output_sequence_length=vocab_config["seq_len"],
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.set_vocabulary(vocab_config["vocabulary"])
    print("Engine ready.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# =====================================================
# 4. PREDICTION FUNCTION
# =====================================================

def predict(text: str, top_k: int = 3):
    vec = vectorizer(tf.constant([text]))
    # Direct model call is faster for single samples
    preds = model(vec, training=False)
    preds = [p.numpy() for p in preds]

    nic_probs = preds[0][0]
    div_probs = preds[1][0]

    top_nic_idx = np.argsort(nic_probs)[::-1][:top_k]
    div_idx     = np.argmax(div_probs)

    top_nics = []
    for idx in top_nic_idx:
        code = nic_map[str(idx)]
        label = nic_labels.get(code, "—")
        top_nics.append({
            "nic_code":   code,
            "nic_label":  label,
            "confidence": float(nic_probs[idx]),
        })

    division = div_map[str(div_idx)]
    guidance = UDYAM_GUIDANCE.get(division, {})

    return {
        "top_nics": top_nics,
        "division": division,
        "div_confidence": float(div_probs[div_idx]),
        "guidance": guidance,
    }

# =====================================================
# 5. PRINT REPORT
# =====================================================

def print_report(text: str):
    result = predict(text, top_k=3)
    best   = result["top_nics"][0]
    div    = result["division"]
    guide  = result["guidance"]

    print("\n-----------------------------------------------------------")
    print(f"  Business: {text[:55]}")
    print("-----------------------------------------------------------")
    print(f"  NIC Code    : {best['nic_code']}")
    print(f"  Activity    : {best['nic_label']}")
    print(f"  Sector      : {div}")
    print(f"  Confidence  : {best['confidence']*100:.1f}%")
    print("-----------------------------------------------------------")
    print(f"  Register At : {guide.get('registration', '—')}")
    print(f"  Compliance  : {guide.get('compliance', '—')}")
    print(f"  Schemes     : {', '.join(guide.get('schemes', []))}")
    print("-----------------------------------------------------------")

# =====================================================
# 6. TEST CASES
# =====================================================

TEST_CASES = [
    "i make soap at home",
    "we grow rice",
    "i am a tailor",
]

print("Running quick test cases...")
for t in TEST_CASES:
    print_report(t)

# =====================================================
# 7. INTERACTIVE MODE
# =====================================================

print("\n" + "=" * 56)
print("  Udyam NIC Predictor - INTERACTIVE MODE")
print("  Type 'exit' to quit")
print("=" * 56)

try:
    while True:
        sys.stdout.write("\nEnter business description: ")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nThank you for using Udyam NIC Predictor.")
            break
            
        print_report(user_input)
except KeyboardInterrupt:
    print("\n\nOperation cancelled. Exiting...")
except EOFError:
    print("\n\nInput closed. Exiting...")
