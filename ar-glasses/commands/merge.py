"""CLI command for finding and merging duplicate face clusters."""

import numpy as np

from models import Person
from storage.database import Database
from config import MERGE_SIMILARITY_THRESHOLD


def do_merge(db: Database, keep: Person, discard: Person):
    """Move all embeddings from discard into keep, then delete discard."""
    for emb in db.get_embeddings(discard.person_id):
        db.add_embedding(keep.person_id, emb)
    db.delete_person(discard.person_id)


def merge_clusters_mode(db: Database):
    """Find and interactively merge clusters that look like the same person."""
    people = [p for p in db.get_all_people() if p.embeddings]

    if len(people) < 2:
        print("Need at least 2 clusters with embeddings.")
        return

    means = {
        p.person_id: np.mean([e.vector for e in p.embeddings], axis=0) for p in people
    }

    suggestions = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            va, vb = means[people[i].person_id], means[people[j].person_id]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na == 0 or nb == 0:
                continue
            sim = float(np.dot(va, vb) / (na * nb))
            if sim >= MERGE_SIMILARITY_THRESHOLD:
                suggestions.append((sim, people[i], people[j]))

    suggestions.sort(reverse=True)

    if not suggestions:
        print(
            f"No cluster pairs found with similarity >= {MERGE_SIMILARITY_THRESHOLD}."
        )
        return

    print(
        f"Found {len(suggestions)} candidate merge(s). Higher = more likely the same person.\n"
    )
    merged_ids: set[int] = set()

    for sim, pa, pb in suggestions:
        if pa.person_id in merged_ids or pb.person_id in merged_ids:
            continue

        last_a = pa.last_seen.strftime("%Y-%m-%d %H:%M") if pa.last_seen else "never"
        last_b = pb.last_seen.strftime("%Y-%m-%d %H:%M") if pb.last_seen else "never"
        print(
            f"[sim={sim:.2f}]  A: '{pa.name}' (ID {pa.person_id}, {len(pa.embeddings)} emb, seen {last_a})"
            f"\n         B: '{pb.name}' (ID {pb.person_id}, {len(pb.embeddings)} emb, seen {last_b})"
        )

        choice = (
            input("  Merge? (a=keep A name, b=keep B name, n=skip): ").strip().lower()
        )
        if choice == "a":
            do_merge(db, keep=pa, discard=pb)
            merged_ids.add(pb.person_id)
            print(f"  Merged B into A → '{pa.name}'")
        elif choice == "b":
            do_merge(db, keep=pb, discard=pa)
            merged_ids.add(pa.person_id)
            print(f"  Merged A into B → '{pb.name}'")
        else:
            print("  Skipped.")

    print("\nMerge complete.")
