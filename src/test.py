import csv
from collections import defaultdict

timestamps = []
ids = []

with open("DoS_dataset.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        ts = float(row[0])
        can_id = int(row[1], 16)   # hex → int
        timestamps.append(ts)
        ids.append(can_id)

print("Total frames:", len(timestamps))


start_time = timestamps[0]
end_time = timestamps[-1]
duration = end_time - start_time
print("Capture duration (s):", duration)
count_per_id = defaultdict(int)
for cid in ids:
    count_per_id[cid] += 1
fps_per_id = {
    cid: cnt / duration
    for cid, cnt in count_per_id.items()
}

top_ids = sorted(fps_per_id.items(), key=lambda x: x[1], reverse=True)

print("\nTop IDs by FPS:")
for cid, fps in top_ids[:10]:
    print(f"ID {hex(cid)} → {fps:.2f} fps")

ifs = []

for i in range(len(timestamps) - 1):
    ifs.append(timestamps[i + 1] - timestamps[i])

mean_ifs = sum(ifs) / len(ifs)
min_ifs = min(ifs)

print("\nMean IFS (s):", mean_ifs)
print("Min IFS (s):", min_ifs)

threshold = 50e-6  # 50 microseconds
small_ifs = [d for d in ifs if d < threshold]
ratio = len(small_ifs) / len(ifs)

print(f"IFS < {threshold}s ratio:", ratio)

timestamps_per_id = defaultdict(list)

for ts, cid in zip(timestamps, ids):
    timestamps_per_id[cid].append(ts)

def mean(lst):
    return sum(lst) / len(lst)

per_id_ifs = []

for cid, ts_list in timestamps_per_id.items():
    if len(ts_list) < 2:
        continue
    diffs = [ts_list[i+1] - ts_list[i] for i in range(len(ts_list)-1)]
    per_id_ifs.append((cid, mean(diffs)))

per_id_ifs.sort(key=lambda x: x[1])

print("\nSuspicious IDs (smallest mean IFS):")
for cid, mifs in per_id_ifs[:5]:
    print(f"ID {hex(cid)} → mean IFS = {mifs:.8f}s")
WINDOW = 0.1  # 100 ms
i = 0
n = len(timestamps)

print("\nDetected DoS windows:")

while i < n:
    start = timestamps[i]
    window_ids = []
    j = i

    while j < n and timestamps[j] - start <= WINDOW:
        window_ids.append(ids[j])
        j += 1

    total = len(window_ids)

    if total > 0:
        freq = defaultdict(int)
        for cid in window_ids:
            freq[cid] += 1

        dominant_id, dom_count = max(freq.items(), key=lambda x: x[1])
        dominance = dom_count / total

        if dominance > 0.6 and total > 200:
            print(f"time={start:.3f}s  ID={hex(dominant_id)}  dominance={dominance:.2f}")

    i = j
