# Udyam NIC Code Predictor
## Micro Language Model — Ministry of Industries (MSME Division)
### Built with TensorFlow | Python 3.8+

---

## The Real Problem This Solves

When a small business owner registers on the **Udyam Portal**
(udyamregistration.gov.in), they must select a NIC 2008 code
from **1,297 options**. Most people have no idea which code applies
to them. They pick wrong, or give up entirely.

This MLM solves that problem:

```
User types   →  "i make soap at home"
Model outputs →  NIC 20231 — Manufacture of soap (all forms)
                 Division  : Manufacturing
                 Register  : udyamregistration.gov.in
                 Schemes   : PMEGP, Mudra Yojana, CGTMSE
                 Compliance: GST, Factory License, BIS Certification
```

**Who benefits:**
- Small business owners filing Udyam registration
- Ministry officers helping MSME applicants
- Common Service Centre (CSC) workers assisting rural entrepreneurs

---

## Dataset Source

The training data replicates three real government datasets:

| Source | What It Contains |
|---|---|
| **NIC 2008** (MoSPI) | 1,297 official activity descriptions |
| **MCA Company Master Data** (data.gov.in) | Real business descriptions + NIC codes |
| **Udyam Registration patterns** | How common people describe their business |

Plain-language inputs ("i make soap", "rice mill", "garment stitching")
are mapped to the correct NIC sub-class code, covering all major sectors.

---

## Project Structure

```
udyam_nic_mlm/
│
├── data/
│   └── create_dataset.py    ← builds industries.csv from NIC + MCA patterns
│
├── train_model.py           ← trains dual-output transformer model
├── inference.py             ← predicts NIC code + Udyam guidance
├── gpucheck.py              ← hardware check before training
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python data/create_dataset.py

# 3. Check hardware
python gpucheck.py

# 4. Train the model
python train_model.py

# 5. Run predictions
python inference.py
```

---

## Model Architecture

```
Input: Plain English business description
         ↓
TextVectorization (12,000 vocab)
         ↓
Token Embedding (128-dim) + Position Embedding
         ↓
Transformer Block 1 (4 heads, 256 FFN)
         ↓
Transformer Block 2 (4 heads, 256 FFN)
         ↓
Global Average Pooling
         ↓
Shared Dense (128, ReLU)
         ↙              ↘
NIC Code Output     Division Output
(100+ classes)      (15 sectors)
```

**Total Parameters: ~2,200,000** (well above minimum 1,000)

---

## Sample Output

```
╔══════════════════════════════════════════════════════╗
║     UDYAM NIC CODE ASSESSMENT REPORT                ║
║     Ministry of Industries — MSME Classification   ║
╠══════════════════════════════════════════════════════╣
║  Business Description: i make soap at home          ║
╠══════════════════════════════════════════════════════╣
║  ✅ NIC Code    : 20231                             ║
║  📋 Activity    : Manufacture of soap all forms     ║
║  🏭 Division    : Manufacturing                     ║
║  📊 Confidence  : 94.2%                             ║
╠══════════════════════════════════════════════════════╣
║  🌐 Register At : udyamregistration.gov.in          ║
║  📜 Compliance  : GST, Factory License, BIS Cert    ║
║  💰 Applicable Schemes:                             ║
║     • PMEGP                                         ║
║     • Mudra Yojana                                  ║
║     • CGTMSE (collateral free loan)                 ║
╚══════════════════════════════════════════════════════╝
```

---

## Sectors Covered (15 Divisions)

Agriculture | Manufacturing | Mining | Construction | Trade |
Transport | Hospitality | IT Services | Finance | Real Estate |
Professional | Education | Healthcare | Services | Energy

---

## Why This Matters for the Ministry

The Ministry of Industries tracks economic activity using NIC codes.
Wrong NIC codes = bad data = wrong policy decisions.

This tool improves **data quality at source** — when people register,
they pick the right code the first time.
