

import json
import os
from collections import Counter, defaultdict

# PATHS — adjust if running from a different directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MAPPING_PATH  = os.path.join(SCRIPT_DIR, "phoneme_cluster_mapping.json")
DATASET_PATH  = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_dataset.txt")
OUTPUT_PATH   = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_clustered.txt")
REPORT_PATH   = os.path.join(PROJECT_DIR, "results", "vocab_reduction_report.md")

print("Loading phoneme_cluster_mapping.json ...")
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# Build a simple phoneme → "C{id}" lookup
phoneme_to_cluster = {}
for phoneme, info in mapping.items():
    phoneme_to_cluster[phoneme] = f"C{info['cluster_id']}"

print(f"  Loaded mapping for {len(phoneme_to_cluster)} phonemes")
print(f"  Unique cluster labels: {sorted(set(phoneme_to_cluster.values()), key=lambda x: int(x[1:]))}")

# Identify singletons
cluster_members = defaultdict(list)
for phoneme, info in mapping.items():
    cid = info["cluster_id"]
    if phoneme not in cluster_members[cid]:
        cluster_members[cid].append(phoneme)

singletons = {cid: members[0] for cid, members in cluster_members.items() if len(members) == 1}
print(f"  Singleton clusters: {singletons}")

print(f"\nReading dataset: {DATASET_PATH}")

total_lines     = 0
converted_lines = 0
skipped_lines   = 0
unknown_phonemes = Counter()   # phonemes not in mapping
original_phoneme_counter = Counter()
cluster_counter = Counter()
lang_counter    = Counter()

output_lines = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        line = line.strip()
        if not line:
            skipped_lines += 1
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            skipped_lines += 1
            continue

        left_part  = parts[0].strip()   # "<HI> अँग"
        phonemes   = parts[1].strip().split()  # ["a", "mq", "g", "a"]

        # Extract lang tag
        if left_part.startswith("<") and ">" in left_part:
            end_idx = left_part.index(">") + 1
            lang_tag = left_part[:end_idx]
            lang_counter[lang_tag] += 1
        else:
            lang_tag = "<UNK>"

        # Substitute each phoneme with its cluster label
        clustered = []
        for p in phonemes:
            original_phoneme_counter[p] += 1
            if p in phoneme_to_cluster:
                cluster_label = phoneme_to_cluster[p]
                clustered.append(cluster_label)
                cluster_counter[cluster_label] += 1
            else:
                # Unknown phoneme — keep as-is and log
                unknown_phonemes[p] += 1
                clustered.append(p)

        # Reconstruct line in same format
        output_line = f"{left_part}\t{' '.join(clustered)}"
        output_lines.append(output_line)
        converted_lines += 1

print(f"  Total lines:     {total_lines}")
print(f"  Converted lines: {converted_lines}")
print(f"  Skipped lines:   {skipped_lines}")

if unknown_phonemes:
    print(f"\n  ⚠️  Unknown phonemes (not in mapping): {dict(unknown_phonemes)}")
else:
    print(f"\n  ✅ All phonemes successfully mapped to clusters")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\n✅ Saved clustered dataset: {OUTPUT_PATH}")
print(f"   Lines written: {len(output_lines)}")

original_vocab = sorted(set(original_phoneme_counter.keys()))
cluster_vocab  = sorted(set(cluster_counter.keys()), key=lambda x: int(x[1:]))

original_count = len(original_vocab)
cluster_count  = len(cluster_vocab)
reduction_pct  = (1 - cluster_count / original_count) * 100

print(f"\n{'='*50}")
print(f"VOCAB REDUCTION SUMMARY")
print(f"{'='*50}")
print(f"  Original phoneme count : {original_count}")
print(f"  Cluster count          : {cluster_count}")
print(f"  Reduction              : {reduction_pct:.1f}%")
print(f"  Total phoneme tokens   : {sum(original_phoneme_counter.values()):,}")

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Build cluster detail table
cluster_details = []
for cid in sorted(cluster_members.keys()):
    members = sorted(cluster_members[cid])
    label = mapping[members[0]]["cluster_label"]
    is_singleton = "⚠️ Singleton" if len(members) == 1 else ""
    total_freq = sum(mapping[m].get("frequency", 0) for m in members)
    cluster_details.append({
        "cid": cid,
        "label_name": label,
        "members": members,
        "member_count": len(members),
        "total_freq": total_freq,
        "singleton": is_singleton,
    })

report = f

for lang, count in sorted(lang_counter.items()):
    report += f"| {lang} | {count:,} |\n"

report += f

for cd in cluster_details:
    members_str = ", ".join(cd["members"])
    report += (
        f"| C{cd['cid']} | {cd['label_name']} | {members_str} | "
        f"{cd['member_count']} | {cd['total_freq']:,} | {cd['singleton']} |\n"
    )

report += f

total_tokens = sum(original_phoneme_counter.values())
for cid, phoneme in sorted(singletons.items()):
    freq = mapping[phoneme].get("frequency", 0)
    pct = (freq / total_tokens * 100) if total_tokens > 0 else 0
    report += f"| C{cid} | `{phoneme}` | {freq:,} | {pct:.3f}% |\n"

report += f

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n✅ Saved vocab reduction report: {REPORT_PATH}")
print(f"\n{'='*50}")
print("TASK 1 COMPLETE")
print(f"{'='*50}")
