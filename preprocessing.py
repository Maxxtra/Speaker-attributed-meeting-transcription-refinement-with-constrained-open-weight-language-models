# make_ami_manifest_smoke
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


AMI_ROOT = Path("/media/user/New Volume/datasets/AMI")
AUDIO_ROOT = AMI_ROOT / "amicorpus"
ANNOT_ROOT = AMI_ROOT / "ami_public_manual_1.6.2"

OUTPUT_CSV = AMI_ROOT / "ami_manifest_smoke.csv"

MAX_MEETINGS = 5
GAP_THRESHOLD = 0.8


def clean_tag(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def find_words_dir():
    candidates = [
        ANNOT_ROOT / "words",
        ANNOT_ROOT / "ami_public_manual_1.6.2" / "words",
        ANNOT_ROOT / "corpusResources" / "words",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = list(ANNOT_ROOT.rglob("words"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find AMI words directory under {ANNOT_ROOT}")


def parse_word_file(xml_path):
    """
    Parses one AMI words XML file, e.g. ES2002a.A.words.xml.
    Speaker is inferred from the filename suffix: A/B/C/D.
    """
    name = xml_path.name
    m = re.match(r"(.+?)\.([A-Z])\.words\.xml$", name)
    if not m:
        return []

    meeting_id = m.group(1)
    speaker_raw = m.group(2)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    words = []

    for elem in root.iter():
        tag = clean_tag(elem.tag)

        if tag != "w":
            continue

        text = "".join(elem.itertext()).strip()
        if not text:
            continue

        start = elem.attrib.get("starttime")
        end = elem.attrib.get("endtime")

        if start is None or end is None:
            continue

        try:
            start = float(start)
            end = float(end)
        except ValueError:
            continue

        words.append({
            "meeting_id": meeting_id,
            "speaker_raw": speaker_raw,
            "start": start,
            "end": end,
            "word": text,
        })

    return words


def build_segments(words):
    """
    Builds approximate speaker turns from word-level AMI annotations.
    Good enough for smoke tests. For final paper, we can improve with official segment annotations.
    """
    words = sorted(words, key=lambda x: (x["start"], x["end"]))

    speaker_map = {}
    segments = []

    current = None

    for w in words:
        raw = w["speaker_raw"]

        if raw not in speaker_map:
            speaker_map[raw] = f"SPEAKER_{len(speaker_map):02d}"

        speaker = speaker_map[raw]

        if current is None:
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": speaker,
                "words": [w["word"]],
            }
            continue

        same_speaker = current["speaker"] == speaker
        small_gap = w["start"] - current["end"] <= GAP_THRESHOLD

        if same_speaker and small_gap:
            current["end"] = max(current["end"], w["end"])
            current["words"].append(w["word"])
        else:
            segments.append(current)
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": speaker,
                "words": [w["word"]],
            }

    if current is not None:
        segments.append(current)

    final_segments = []
    lines = []

    for s in segments:
        text = " ".join(s["words"]).strip()
        if not text:
            continue

        final_segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "speaker": s["speaker"],
            "text": text,
        })

        lines.append(f'{s["speaker"]}: {text}')

    transcript = "\n".join(lines)
    return transcript, final_segments


def main():
    words_dir = find_words_dir()
    print(f"Using words dir: {words_dir}")

    audio_files = sorted(AUDIO_ROOT.glob("*/audio/*.Mix-Headset.wav"))

    rows = []

    for audio_path in audio_files:
        meeting_id = audio_path.parent.parent.name

        word_files = sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))

        if not word_files:
            print(f"[WARN] No word files for {meeting_id}")
            continue

        all_words = []
        for wf in word_files:
            all_words.extend(parse_word_file(wf))

        if not all_words:
            print(f"[WARN] No parsed words for {meeting_id}")
            continue

        text_reference, segments = build_segments(all_words)

        rows.append({
            "audio_number": meeting_id,
            "audio_path": str(audio_path),
            "text_reference": text_reference,
            "speaker_segments_reference": json.dumps([
                {
                    "start": s["start"],
                    "end": s["end"],
                    "speaker": s["speaker"],
                    "text": s["text"],
                }
                for s in segments
            ]),
            "num_reference_segments": len(segments),
            "num_reference_words": len(all_words),
        })

        print(f"[OK] {meeting_id}: words={len(all_words)}, segments={len(segments)}")

        if len(rows) >= MAX_MEETINGS:
            break

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(df[["audio_number", "audio_path", "num_reference_segments", "num_reference_words"]])
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()