# Speaker-Attributed Meeting Transcription Refinement with Constrained Open-Weight Language Models

Reproducibility repository for the FGCS paper. The pipeline combines Whisper (ASR) + Pyannote (speaker diarization) with a constrained Qwen2.5 LLM refinement stage, evaluated on the AMI Meeting Corpus across three audio conditions.

---

## Repository structure

```
.
├── amiBuild-*.wget.sh            # Script to download the AMI audio corpus (~172 GB)
├── ami_public_manual_1.6.2/     # AMI manual annotations (ground truth transcriptions)
├── ami_manifest_smoke.csv       # Quick smoke-test manifest (5 meetings)
├── preprocessing.py             # Builds meeting manifests from annotations + audio
├── dataset_interpretation.py    # Dataset statistics (duration, overlap, speakers)
├── FGCS_mix_headset_raw.ipynb   # Experiment: Mix-Headset condition
├── FGCS_individual_headsets_raw.ipynb  # Experiment: Individual Headsets condition
├── FGCS_microphone_array_raw.ipynb     # Experiment: Microphone Array condition
├── work/                        # Generated manifests and pipeline outputs (CSVs)
├── .env.example                 # Template for environment variables
└── .gitignore
```

---

## Reproducing the experiments

### 1. Clone this repository

```bash
git clone https://github.com/Maxxtra/Speaker-attributed-meeting-transcription-refinement-with-constrained-open-weight-language-models.git
cd Speaker-attributed-meeting-transcription-refinement-with-constrained-open-weight-language-models
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your HuggingFace token and any path overrides
```

You need a HuggingFace account and must accept the license for:
- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (or 3B variant)

### 3. Install Python dependencies

```bash
pip install soundfile pandas pydub transformers accelerate bitsandbytes sentencepiece \
            openai-whisper pyannote.audio torch torchvision torchaudio \
            jiwer faster-whisper
```

### 4. Download the AMI audio corpus (~172 GB)

The audio is not stored in this repository. Run the provided wget script from the repo root:

```bash
bash amiBuild-*.wget.sh
```

This downloads all AMI meeting audio into `amicorpus/` (Mix-Headset, Individual Headsets, Microphone Array WAV files). The download script was generated from the [AMI Corpus Mirror](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/).

The manual annotations (`ami_public_manual_1.6.2/`) are already included in this repository.

### 5. Build the meeting manifests

Update the path constants at the top of `preprocessing.py` and `dataset_interpretation.py` to point to your local clone, then run:

```bash
python preprocessing.py          # builds ami_manifest_smoke.csv (5 meetings, quick test)
python dataset_interpretation.py # computes corpus statistics → work/
```

For the full manifests used in the notebooks, run the manifest-building cells inside each notebook (they write to `work/ami_manifest_full_*.csv`).

### 6. Run the experiment notebooks

Open and run the notebooks in order:

| Notebook | Audio condition | Output prefix |
|---|---|---|
| `FGCS_mix_headset_raw.ipynb` | Mix-Headset | `work/*_mix_headset_*` |
| `FGCS_individual_headsets_raw.ipynb` | Individual Headsets | `work/*_individual_headsets_*` |
| `FGCS_microphone_array_raw.ipynb` | Microphone Array | `work/*_microphone_array_*` |

Each notebook:
1. Loads the manifest
2. Runs Whisper transcription + Pyannote diarization (baseline)
3. Optionally runs Qwen2.5 LLM refinement (`RUN_LOCAL_LLM=true`)
4. Computes WER / cpWER / DER metrics
5. Saves results to `work/`

Pre-computed results for the smoke test and the full Mix-Headset + 1-meeting runs are included in `work/` for quick inspection without re-running the full pipeline.

---

## Hardware requirements

| Stage | Minimum |
|---|---|
| Whisper `base.en` | CPU (slow) or any GPU |
| Pyannote diarization | 4 GB VRAM |
| Qwen2.5-7B-Instruct (int4) | ~8 GB VRAM |
| Qwen2.5-3B-Instruct (int4) | ~4 GB VRAM |

Full corpus inference (all conditions) takes ~24–48 h on a single A100/4090.

---

## Data

- **AMI Meeting Corpus** — [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/)  
  License: [Creative Commons Attribution 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)  
  The annotations in `ami_public_manual_1.6.2/` are the official AMI manual annotations v1.6.2.

- **Whisper** — OpenAI, MIT License
- **Pyannote** — MIT License (model weights require HuggingFace license acceptance)
- **Qwen2.5** — Tongyi Qianwen License

---

## Citation

If you use this code or results, please cite the corresponding FGCS paper (citation to be added upon publication).
