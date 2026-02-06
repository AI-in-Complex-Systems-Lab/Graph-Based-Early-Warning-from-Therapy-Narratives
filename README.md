# Graph-Based Early Warning from Therapy Narratives  
### A System-of-Systems Copilot for Behavioral Health (Research Prototype)

This repository contains a **research prototype** of a clinician-facing AI “copilot” designed to support **early warning detection** from therapy-style narratives. 
The system is intentionally built to **augment—not replace—clinical judgment** by turning session transcripts into **temporally ordered sentence graphs** and producing **auditable, clinician-legible analytics**.

The pipeline couples:

- **Sentence-level emotion annotation** (upstream annotator; each sentence gets an emotion label)
- **Session-level status prediction** using a **Graph Neural Network (GATv2)** over a **sentence graph** (each session is a graph; each sentence is a node)
- **Clinician-facing visualizations** that surface *why* a signal appears (polarity→topic flows, longitudinal “personal-stories” graphs)

> **Important:** The included dataset is **synthetic** (AI-generated therapy-style narratives) and is provided for experimentation and stress-testing only. 
It is **not** clinical-grade, not validated for real-world use, and must not be used for diagnosis or autonomous decision-making.

---

## Key Idea

Each therapy visit is represented as a **graph**:
- **Nodes:** sentences (with emotion labels + topic + position)
- **Edges:** temporal adjacency
- **Graph label:** session status `{improving, stable, deteriorating}`

The GNN predicts **session-level trajectory status** and supports rapid clinician audit by linking model outputs back to specific sentences/topics.

---

## Repository Structure
├── Data/
│ ├── narratives.csv
│ └── scores.csv
└── Code/
├── 1.ipynb
└── 2.ipynb


### `Data/`

#### `Data/narratives.csv`
Synthetic therapy narratives in a session-by-session structure.

**Expected columns:**
- `patient_id` : patient identifier
- `session_id` : session identifier (or session number)
- `date` : session date
- `sentence` : sentence text (one row per sentence)
- `emotion` : sentence-level emotion label
- `topic` : topic tag for the sentence

#### `Data/scores.csv`
Session-level synthetic self-report outcomes aligned to PHQ-9 / GAD-7 style scoring.

**Expected columns:**
- `patient_id`
- `session_id`
- `phq9_total` : 0–27
- `gad7_total` : 0–21

---

## `Code/`

`Code/1.ipynb` — Training & Evaluation Notebook

`Code/2.ipynb` — Interface / Dashboard Notebook

---

## Notes on Safety, Ethics, and Intended Use

- This project is a **research prototype** for exploring **graph-based modeling** and **auditable analytics** over therapy-style text.
- The dataset is **synthetic** and does not represent real patient records.
- The system must **never auto-escalate** or replace clinical judgment.

---

## Citation

If you build on this work, please cite the paper:

**Graph-Based Early Warning from Therapy Narratives: A System-of-Systems Copilot for Behavioral Health**

(We will provide the link when available.)


