# Speaker-Attributed Meeting Transcription Refinement with Constrained Open-Weight Language Models

> **Published in:** Future Generation Computer Systems, Volume 185, December 2026, 108648  
> **DOI:** https://doi.org/10.1016/j.future.2026.108648  
> **License:** Open access, Creative Commons

**Authors:** Costin-Alexandru Deonise, Taisia-Maria Coconu, Muhammad Khurram Zahur Bajwa, Catalin Negru, Bogdan-Costel Mocanu, Aniello Castiglione, Florin Pop

---

## Abstract

Speaker diarization and Automatic Speech Recognition (ASR) are traditionally evaluated as independent components. This paper presents a **modular offline pipeline** for speaker-attributed meeting transcription that integrates:
- **Pyannote 3.1** — speaker diarization
- **Whisper** — ASR (base.en for full corpus evaluation)
- **Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct** — constrained LLM post-processing for transcript refinement

Evaluation is performed on the **AMI Meeting Corpus** (143 meetings, 77.63 h) using DER, JER, WER, and cpWER. The key finding is that open-weight LLMs are more effective as **selective verification modules** than as autonomous rewriters in high-fidelity transcription systems.

---

## Repository structure

```
.
├── amiBuild-*.wget.sh                   # Downloads the AMI audio corpus (~172 GB)
├── ami_public_manual_1.6.2/             # AMI manual annotations v1.6.2 (ground truth)
│   ├── words/                           # Word-level XML transcriptions (687 files)
│   ├── segments/                        # Segment-level XML annotations
│   └── ...
├── ami_manifest_smoke.csv               # Smoke-test manifest (5 meetings)
├── preprocessing.py                     # Builds meeting manifests from annotations + audio
├── dataset_interpretation.py            # Corpus statistics (duration, overlap, speakers)
├── FGCS_mix_headset_raw.ipynb           # Experiment: Mix-Headset condition (primary)
├── FGCS_individual_headsets_raw.ipynb   # Experiment: Individual Headsets condition
├── FGCS_microphone_array_raw.ipynb      # Experiment: Microphone Array condition
├── work/                                # Generated manifests and pipeline outputs (CSVs)
├── .env.example                         # Template for environment variables
└── .gitignore
```

---

## Results

### Baseline: Pyannote–Whisper (143 AMI meetings, Headset Mix)

| Metric | Value |
|---|---|
| WER | 0.802 |
| cpWER | 0.790 |
| DER | 0.160 |
| JER | 0.209 |
| Purity | 0.911 |
| Coverage | 0.886 |
| Runtime | 652.65 min |

*Pyannote/speaker-diarization-3.1 + Whisper base.en, no LLM post-processing.*

### LLM post-processing (normalized transcript-level results)

| Method | Model / setup | Filtering | norm-WER ↓ | norm-cpWER ↓ | Token F1 ↑ | Rep. reduction |
|---|---|---|---|---|---|---|
| Baseline | None | — | 0.50 | 0.52 | 0.86 | 0.00 |
| LLM | Qwen2.5-3B zero-shot | No | 0.82 | 0.83 | 0.42 | 0.75 |
| LLM | Qwen2.5-3B one-shot | No | 0.70 | 0.70 | 0.59 | 0.51 |
| LLM | Qwen2.5-7B zero-shot | No | 0.62 | 0.65 | 0.70 | 0.40 |
| LLM | Qwen2.5-7B one-shot | No | 0.55 | 0.58 | 0.78 | 0.35 |
| **LLM** | **Qwen2.5-7B one-shot** | **Yes** | **0.49** | **0.51** | **0.85** | **0.22** |

*Filtered = conservative validation that rejects outputs with excessive lexical changes or invalid speaker tags.*

---

## Reproducing the experiments

### 1. Clone this repository

```bash
git clone https://github.com/Maxxtra/Speaker-attributed-meeting-transcription-refinement-with-constrained-open-weight-language-models.git
cd Speaker-attributed-meeting-transcription-refinement-with-constrained-open-weight-language-models
```

### 2. Environment setup

The paper used **Python 3.10.12** with **CUDA 11.6.0 / cuDNN** via Anaconda:

```bash
conda create -n ami-pipeline python=3.10.12
conda activate ami-pipeline
pip install soundfile pandas pydub \
            openai-whisper faster-whisper \
            pyannote.audio \
            transformers accelerate bitsandbytes sentencepiece \
            torch torchvision torchaudio \
            jiwer
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env — fill in your HuggingFace token and optionally adjust paths
```

You need a HuggingFace account and must accept the gated-model license for:
- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

### 4. Download the AMI audio corpus (~172 GB)

The audio is not stored in this repository. Run the provided wget script from the repo root:

```bash
bash amiBuild-*.wget.sh
```

This populates `amicorpus/` with WAV files for all three audio conditions (Mix-Headset, Individual Headsets, Microphone Array). Source: [AMI Corpus Mirror](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/).

The manual annotations in `ami_public_manual_1.6.2/` are already included in this repository.

### 5. Build the meeting manifests

Update the path constants at the top of `preprocessing.py` and `dataset_interpretation.py` to point to your local clone, then:

```bash
python preprocessing.py           # builds ami_manifest_smoke.csv (5 meetings, quick sanity check)
python dataset_interpretation.py  # computes corpus statistics → work/
```

Full manifests for all 143 meetings are generated by the manifest-building cells inside each notebook (they write to `work/ami_manifest_full_*.csv`).

### 6. Run the experiment notebooks

| Notebook | Audio condition | Description |
|---|---|---|
| `FGCS_mix_headset_raw.ipynb` | **Mix-Headset** (primary) | One mixed close-talking stream per meeting |
| `FGCS_individual_headsets_raw.ipynb` | Individual Headsets | Separate close-talking track per speaker |
| `FGCS_microphone_array_raw.ipynb` | Microphone Array | Far-field table-top microphones |

Each notebook:
1. Loads the meeting manifest
2. Runs **Pyannote** diarization (constrained to 4–5 speakers per meeting)
3. Runs **Whisper base.en** transcription on each diarized segment
4. Optionally runs **Qwen2.5** LLM refinement (`RUN_LOCAL_LLM=true` in `.env`)
5. Validates LLM output (speaker-tag format + lexical fidelity check)
6. Computes WER, cpWER, DER, JER, Purity, Coverage
7. Saves results to `work/`

Pre-computed results for the smoke test and the 1-meeting run are included in `work/` for immediate inspection.

---

## Pipeline overview

```
Audio (.wav)  +  AMI annotations (.xml)
        │
        ▼
[1] Parse word-level XML → reference transcript + speaker activity segments
        │
        ▼
[2] Pyannote 3.1 → speaker diarization (RTTM segments)
        │
        ▼
[3] Whisper base.en → ASR per diarized segment
        │
        ▼
[4] Build speaker-attributed transcript: "SPEAKER_XX: text"
        │
        ▼
[5] Qwen2.5 LLM (optional) → constrained speaker-label correction (JSON)
        │
[6] Validation: reject if speaker tags malformed OR lexical change > threshold
        │
        ▼
[7] Evaluate: WER, cpWER, DER, JER, Purity, Coverage
```

---

## Hardware requirements

| Component | Used in paper | Minimum for base.en |
|---|---|---|
| GPU (diarization) | NVIDIA RTX 3060 (12 GB VRAM) | 4 GB VRAM |
| GPU (Whisper base.en) | RTX 3060 | CPU possible (slow) |
| GPU (Qwen2.5-7B int4) | RTX 4090 | ~8 GB VRAM |
| GPU (Qwen2.5-3B int4) | — | ~4 GB VRAM |

Full baseline evaluation (143 meetings, Headset Mix, no LLM) took **652.65 min** on the above hardware.

---

## Dataset

**AMI Meeting Corpus** — [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/)

| Audio condition | Meetings | Files | Meeting h | Channel h | Avg. speakers | Ref. words | Overlap |
|---|---|---|---|---|---|---|---|
| Headset Mix | 143 | 143 | 77.63 | 77.63 | 4.02 | 732,982 | 11.85% |
| Individual Headsets | 143 | 575 | 77.63 | 314.10 | 4.02 | 732,982 | 11.85% |
| Microphone Array | 143 | 2189 | 77.63 | 1204.57 | 4.02 | 732,982 | 11.85% |

AMI Corpus License: [Creative Commons Attribution 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)  
The annotations in `ami_public_manual_1.6.2/` are the official AMI manual annotations v1.6.2.

---

## Acknowledgments

This work was partially supported by the project HRIA: Romanian Hub for Artificial Intelligence, Smart Growth, Digitization and Financial Instruments Program, 2021–2027, MySMIS no. 334906 and DACISLab: Virtual Laboratory on Open Data and Open Science in the New Generation of Continuum Computing Systems, project number PN-IV-PCB-RO-MD-2024-0364, within PNCDI IV.

---

## Citation

```bibtex
@article{deonise2026speaker,
  title     = {Speaker-attributed meeting transcription refinement with constrained open-weight language models},
  author    = {Deonise, Costin-Alexandru and Coconu, Taisia-Maria and Bajwa, Muhammad Khurram Zahur and Negru, Catalin and Mocanu, Bogdan-Costel and Castiglione, Aniello and Pop, Florin},
  journal   = {Future Generation Computer Systems},
  volume    = {185},
  pages     = {108648},
  year      = {2026},
  month     = {December},
  publisher = {Elsevier},
  doi       = {10.1016/j.future.2026.108648}
}
```
