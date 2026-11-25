# Multimodal Depression Detection on DAIC-WOZ

Session-level, leakage-controlled depression detection from interview **text, voice, face, gaze and head movement** on the DAIC-WOZ corpus. The pipeline trains LSTM and BiLSTM text models on three input representations, compares them with three pretrained language models, and fuses the text score with voice, face, gaze and behaviour features.

This repository accompanies the course-project article *Multimodal Depression Detection from Speech, Facial and Text Streams on the DAIC-WOZ Corpus* (DSC 501, SRM University AP).

> **Research use only.** This is not a clinical tool and offers no diagnostic validity.

---

## Contents

| Notebook | Purpose | Status |
|---|---|---|
| `Multimodal_CEDS_v1.ipynb` | First exploration of the two case-study sessions (303 and 319): transcript, sentiment, voice, face (KMeans + HMM), gaze and head pose, simple fusion | Early draft, kept for history |
| `Multimodal_CEDS_v2.ipynb` | Rewrite with corrected COVAREP column names, per-answer sentiment, Ellie's question attached to each answer, leakage-safe LSTM/BiLSTM training, three pretrained models, report figures | Intermediate |
| `Multimodal_CEDS_v3.ipynb` | **Final notebook.** Part A explores the case sessions. Part B runs the full cohort pipeline: stratified 80/10/10 session split, three text representations, out-of-fold scoring, shuffled-label control, multimodal features, fusion, ablation, report figures and tables | **Use this one** |

---

## Data (not included)

DAIC-WOZ is distributed by the USC Institute for Creative Technologies under a data use agreement. **This repository contains no DAIC-WOZ files and no participant information.**

1. Request access at <https://dcapswoz.ict.usc.edu/>.
2. Download the per-session zip files (for example `303_P.zip`) and the AVEC 2017 label files.
3. Put them in one folder on Google Drive:

```
MyDrive/Multimodal_CEDS/
├── 303_P.zip   319_P.zip   320_P.zip   ...   (one zip per session)
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
└── full_test_split.csv
```

Each session provides a transcript, COVAREP voice features, formant tracks, CLNF action units, gaze, head pose and audio. The notebook extracts only the files it needs.

---

## Quick start (Google Colab)

1. Open `Multimodal_CEDS_v3.ipynb` in Colab.
2. Mount Drive and unzip the sessions (the first cells do this into `/content/daic`).
3. Install the extra packages:
   ```bash
   pip install -q gensim
   python -m spacy download en_core_web_sm
   ```
4. Run **Section 1 (Setup and configuration)** first. Edit only this block to change sessions, folders or the cohort list (`TRAIN_SOURCES`).
5. Run the rest in order. Part B (Sections 9 to 16) needs only Section 1 and runs on its own.

### Notebook map (v3)

| Section | What it does |
|---|---|
| 1 Setup | libraries, paths, sessions, cohort list, seed |
| 2 to 4 | case-session transcripts, Ellie's question for each answer, PHQ-8 items, tone, distinctive wording, symptom evidence |
| 5 to 8 | voice, face, gaze and head descriptors, per-minute fusion table (Part A, sessions 303 and 319) |
| 9 Cohort and split | unzip needed files, labels, stratified 80/10/10 split by session |
| 10 Text data | participant-only windows, raw tokens, Word2Vec, Word2Vec with positional encoding |
| 11 Text models | LSTM and BiLSTM, out-of-fold scores, best model, shuffled-label control |
| 12 Multimodal features | numeric voice, face, gaze, head and behaviour features per session |
| 13 Fusion | text score plus features, scikit-learn classifier, ablations, final decision |
| 14 to 15 | modelling figures, every figure and table saved and zipped |
| 16 Appendix | pretrained reference scores on the case sessions |

---

## Method in short

- **Label:** PHQ-8 binary of the whole interview (score of 10 or more is depressed), from the AVEC split files. Text windows inherit their session's label.
- **Text:** participant turns only, cut into 60-word windows (minimum 20). Windows sharing an 8-gram with interviewer text are removed.
- **Text models:** LSTM and BiLSTM (hidden size 64, embedding 100, dropout 0.4, Adam 1e-3, weight decay 1e-4, batch 64, at most 15 epochs, patience 3, gradient clip 1, class-weighted BCE) on three representations. A TF-IDF + logistic regression model serves as a reference.
- **Pretrained comparators (used as is, never fitted):** `rafalposwiata/deproberta-large-depression`, `cardiffnlp/twitter-roberta-base-sentiment-latest` (negative probability), `facebook/bart-large-mnli` (zero-shot).
- **Fusion:** median imputer, scaler, then logistic regression (C = 0.3, balanced) or random forest (300 trees, depth 3). The feature set and classifier are chosen by repeated stratified cross-validation on training sessions only. The decision threshold is fixed at 0.5.

### Leakage controls

1. Splitting is by **session**, stratified by label. The notebook asserts that every session is in exactly one partition. Sessions 303 and 319 stay outside the cohort as untouched case studies.
2. Vocabulary, Word2Vec and model weights are fitted on training windows only, including inside every cross-validation fold.
3. Validation sessions only decide when training stops. Test sessions are used for reporting only.
4. The fusion model receives an **out-of-fold** text score for each training session.
5. Imputation and scaling are fitted inside each fold.
6. A shuffled-label control shows what the pipeline scores when there is nothing to learn.
7. No pretrained model output is ever used as a label, feature or fusion input.

---

## Results (24-session cohort, seed 42)

Cohort: 24 sessions (15 depressed, 9 not depressed), split 18 / 3 / 3 into train, validation and test, plus two case sessions.

| Finding | Value |
|---|---|
| Best trained text model (raw tokens + BiLSTM), out-of-fold session AUC | 0.545 |
| Pretrained models, session AUC on training sessions | 0.649 to 0.701 |
| Shuffled-label control, session AUC (real labels / shuffled mean) | 0.250 / 0.417 |
| Best fusion set: six speech-behaviour features, cross-validated AUC | 0.961 |
| Held-out sessions classified correctly (6 sessions) | 5 of 6 (majority class: 4 of 6) |

**Read these numbers with care.** Validation and test each hold three sessions, so one session moves a metric a lot. The text models did not beat chance. In this cohort the label is perfectly ordered by session number (depressed sessions 320 to 355, non-depressed 457 to 491), so recording conditions may be confounded with the label. Both case sessions were classified wrongly.

---

## Tested environment

Google Colab, CPU only (Intel Xeon 2.20 GHz, 2 cores, 12 GB RAM), Linux 6.6.159.

| Package | Version |
|---|---|
| Python | 3.13 |
| PyTorch (CPU) | 2.14.1 |
| gensim | 4.4.0 |
| scikit-learn | 1.6.1 |
| transformers | 5.22.1 |
| pandas / NumPy / Matplotlib | 2.2.3 / 2.4.6 / 3.10.0 |
| also used | seaborn, scipy, spaCy |

`requirements.txt` (create it from the table above):

```
torch
gensim
scikit-learn
transformers
pandas
numpy
matplotlib
seaborn
scipy
spacy
```

The first notebook (v1) also uses `vaderSentiment`, `hmmlearn`, `nltk` and `wordcloud`.

---

## Reproducing the report

Run Section 15 of `Multimodal_CEDS_v3.ipynb`. It redraws every registered figure, writes each table as CSV and zips them. All figures use fixed colours and `seed = 42`.

---

## Limitations

- Only 24 sessions, so held-out metrics are anecdotal.
- Best of 14 configurations is chosen on the same 18 training sessions, so cross-validated scores are optimistic.
- Label ordering by session number may confound recording conditions with depression.
- No subgroup analysis (for example by gender).
- One corpus, one protocol, no raw video.

---

## Citation and contact

If you use this work, please cite the accompanying article and the DAIC-WOZ corpus (Gratch et al., 2014).

**Author:** Soham Mandal, Department of Computer Science and Engineering, SRM University AP.

## License

Add a license of your choice (for example MIT for the code). The DAIC-WOZ data are **not** covered by it and remain under the USC ICT data use agreement.
