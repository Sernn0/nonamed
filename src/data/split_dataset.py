import os
import json
import random

INDEX_PATH = "data/handwriting_processed/handwriting_index.json"
OUT_TRAIN = "data/handwriting_processed/handwriting_index_train.json"
OUT_VAL = "data/handwriting_processed/handwriting_index_val.json"
OUT_TEST = "data/handwriting_processed/handwriting_index_test.json"

RANDOM_SEED = 42

# 비율로 나눌 경우 (원하면 writer 수로 직접 나눠도 됨)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def group_by_writer(entries):
    """
    entries: list of dict
    returns: dict writer_id -> list of entries
    """
    grouped = {}
    for e in entries:
        # writer_id가 있다고 가정, 없으면 e["writer"]를 써도 됨
        writer_id = e.get("writer_id")
        if writer_id is None:
            # fallback
            writer_id = e["writer"]
        if writer_id not in grouped:
            grouped[writer_id] = []
        grouped[writer_id].append(e)
    return grouped


def split_writers(writer_ids):
    writer_ids = list(writer_ids)
    random.Random(RANDOM_SEED).shuffle(writer_ids)

    n = len(writer_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # 나머지는 test로
    n_test = n - n_train - n_val

    train_writers = writer_ids[:n_train]
    val_writers = writer_ids[n_train : n_train + n_val]
    test_writers = writer_ids[n_train + n_val :]

    return train_writers, val_writers, test_writers


def flatten_by_writers(grouped, writer_subset):
    out = []
    for w in writer_subset:
        out.extend(grouped[w])
    return out


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}  (samples: {len(data)})")


def main():
    print("=== Loading master index ===")
    entries = load_index(INDEX_PATH)
    print(f"Total samples: {len(entries)}")

    print("=== Grouping by writer ===")
    grouped = group_by_writer(entries)
    writer_ids = sorted(grouped.keys())
    print(f"Total writers: {len(writer_ids)}")

    print("=== Splitting writers (train/val/test) ===")
    train_writers, val_writers, test_writers = split_writers(writer_ids)

    print(f"Train writers: {len(train_writers)}")
    print(f"Val writers:   {len(val_writers)}")
    print(f"Test writers:  {len(test_writers)}")

    train_entries = flatten_by_writers(grouped, train_writers)
    val_entries = flatten_by_writers(grouped, val_writers)
    test_entries = flatten_by_writers(grouped, test_writers)

    print(f"Train samples: {len(train_entries)}")
    print(f"Val samples:   {len(val_entries)}")
    print(f"Test samples:  {len(test_entries)}")

    print("=== Saving split index files ===")
    save_json(OUT_TRAIN, train_entries)
    save_json(OUT_VAL, val_entries)
    save_json(OUT_TEST, test_entries)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
