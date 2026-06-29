from pathlib import Path
from collections import defaultdict
import re
import json
import soundfile as sf
import pandas as pd
import xml.etree.ElementTree as ET

AMI_ROOT = Path("/media/user/New Volume/datasets/AMI")
AUDIO_ROOT = AMI_ROOT / "amicorpus"
WORDS_DIR = AMI_ROOT / "ami_public_manual_1.6.2" / "words"
WORK_ROOT = AMI_ROOT / "work"

def local_name(tag):
    return tag.split("}")[-1] if "}" in tag else tag

def safe_duration(path):
    try:
        return sf.info(str(path)).duration
    except Exception:
        return None

def parse_reference_segments(meeting_id):
    word_files = sorted(WORDS_DIR.glob(f"{meeting_id}.*.words.xml"))
    segments = []

    for wf in word_files:
        m = re.match(r"(.+?)\.([A-Z])\.words\.xml$", wf.name)
        if not m:
            continue

        speaker_raw = m.group(2)

        try:
            tree = ET.parse(wf)
        except Exception:
            continue

        root = tree.getroot()

        for elem in root.iter():
            if local_name(elem.tag) != "w":
                continue

            start = elem.attrib.get("starttime")
            end = elem.attrib.get("endtime")
            text = " ".join("".join(elem.itertext()).split())

            if not start or not end or not text:
                continue

            try:
                start = float(start)
                end = float(end)
            except ValueError:
                continue

            if end > start:
                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker_raw,
                    "word": text,
                })

    return segments

def union_and_overlap_time(intervals):
    """
    Returns:
    - union speech time
    - overlapped speech time, where at least 2 speakers are active
    """
    events = []

    for start, end in intervals:
        if end > start:
            events.append((start, 1))
            events.append((end, -1))

    if not events:
        return 0.0, 0.0

    events.sort()
    active = 0
    last_t = None
    union = 0.0
    overlap = 0.0

    for t, delta in events:
        if last_t is not None and t > last_t:
            duration = t - last_t
            if active >= 1:
                union += duration
            if active >= 2:
                overlap += duration

        active += delta
        last_t = t

    return union, overlap

def summarize_condition(condition_name, meeting_to_files, annotated_meetings_only=True):
    rows = []

    for meeting_id, files in sorted(meeting_to_files.items()):
        ref_segments = parse_reference_segments(meeting_id)

        if annotated_meetings_only and not ref_segments:
            continue

        durations = []
        for f in files:
            d = safe_duration(f)
            if d is not None:
                durations.append(d)

        if not durations:
            continue

        speakers = sorted(set(s["speaker"] for s in ref_segments))
        intervals = [(s["start"], s["end"]) for s in ref_segments]

        speech_union_s, overlap_s = union_and_overlap_time(intervals)

        ref_start = min((s["start"] for s in ref_segments), default=None)
        ref_end = max((s["end"] for s in ref_segments), default=None)
        ref_span_s = (ref_end - ref_start) if ref_start is not None and ref_end is not None else None

        rows.append({
            "condition": condition_name,
            "meeting_id": meeting_id,
            "num_files": len(files),
            "meeting_duration_s": max(durations),
            "channel_duration_s": sum(durations),
            "num_speakers": len(speakers),
            "num_reference_words": len(ref_segments),
            "reference_span_s": ref_span_s,
            "speech_union_s": speech_union_s,
            "overlap_s": overlap_s,
            "overlap_ratio_in_speech": overlap_s / speech_union_s if speech_union_s > 0 else None,
        })

    return pd.DataFrame(rows)

meeting_to_mix = defaultdict(list)
meeting_to_indiv = defaultdict(list)
meeting_to_array = defaultdict(list)

for meeting_dir in sorted(AUDIO_ROOT.iterdir()):
    if not meeting_dir.is_dir():
        continue

    meeting_id = meeting_dir.name
    audio_dir = meeting_dir / "audio"

    if not audio_dir.exists():
        continue

    for f in audio_dir.glob("*.Mix-Headset.wav"):
        meeting_to_mix[meeting_id].append(f)

    for f in audio_dir.glob("*.Headset-*.wav"):
        meeting_to_indiv[meeting_id].append(f)

    for f in audio_dir.glob("*.Array*.wav"):
        meeting_to_array[meeting_id].append(f)

df_mix = summarize_condition("Headset mix", meeting_to_mix)
df_indiv = summarize_condition("Individual headsets", meeting_to_indiv)
df_array = summarize_condition("Microphone array", meeting_to_array)

df_all = pd.concat([df_mix, df_indiv, df_array], ignore_index=True)

summary = (
    df_all
    .groupby("condition")
    .agg(
        meetings=("meeting_id", "nunique"),
        files=("num_files", "sum"),
        meeting_hours=("meeting_duration_s", lambda x: x.sum() / 3600),
        channel_hours=("channel_duration_s", lambda x: x.sum() / 3600),
        avg_speakers_per_meeting=("num_speakers", "mean"),
        min_speakers=("num_speakers", "min"),
        max_speakers=("num_speakers", "max"),
        total_reference_words=("num_reference_words", "sum"),
        speech_hours=("speech_union_s", lambda x: x.sum() / 3600),
        overlap_hours=("overlap_s", lambda x: x.sum() / 3600),
        avg_overlap_ratio=("overlap_ratio_in_speech", "mean"),
    )
    .reset_index()
)

summary["meeting_hours"] = summary["meeting_hours"].round(2)
summary["channel_hours"] = summary["channel_hours"].round(2)
summary["speech_hours"] = summary["speech_hours"].round(2)
summary["overlap_hours"] = summary["overlap_hours"].round(2)
summary["avg_speakers_per_meeting"] = summary["avg_speakers_per_meeting"].round(2)
summary["avg_overlap_ratio"] = summary["avg_overlap_ratio"].round(4)

print(summary)

df_all.to_csv(WORK_ROOT / "ami_audio_condition_per_meeting.csv", index=False)
summary.to_csv(WORK_ROOT / "ami_audio_condition_summary.csv", index=False)

print()
print("Saved:")
print(WORK_ROOT / "ami_audio_condition_per_meeting.csv")
print(WORK_ROOT / "ami_audio_condition_summary.csv")