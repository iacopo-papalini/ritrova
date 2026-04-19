"""Dissolve pet 'dumpster' subjects, re-cluster, mint per-cluster strangers.

Motivation (see the Apr 19 conversation on circle architecture): a single
``Cani a caso`` / ``Gatti a caso`` subject groups many different animals,
so its SigLIP centroid averages wildly different features and poisons
merge-suggestions / auto-assign. The fix is to push pets onto the same
Stranger-per-cluster model that works for humans: accept that SigLIP
clusters aren't as identity-clean as ArcFace face clusters (a given cat
may end up as ``StrangerCat #17`` in one pose and ``StrangerCat #103`` in
another), but at least each cluster is a visually-coherent bundle with a
centroid that means something.

Steps:

  1. Find pet subjects whose names match the ``--dissolve-name`` list
     (default: Cani a caso, Gatti a caso).
  2. Null the ``person_id`` on every finding currently pointing at them.
  3. Delete the dumpster subjects (FK cascades remove circle rows and
     orphan the just-unassigned findings).
  4. Re-cluster the pet embeddings at a tighter threshold
     (default 0.18, vs the 0.23 production default) so clusters bias
     toward homogeneity at the cost of more singletons — acceptable per
     user direction (fewer pet photos overall, small clusters OK).
  5. For every resulting unnamed pet cluster, mint ``StrangerDog #N`` or
     ``StrangerCat #N`` (per cluster species), assign the cluster to it,
     and enroll the new subject in the ``Strangers`` circle so the
     hide-strangers filter and the anchor-centroid exclusion both pick
     it up automatically.

Run:

    uv run python scripts/rebuild_stranger_pets.py --dry-run
    uv run python scripts/rebuild_stranger_pets.py --apply
    uv run python scripts/rebuild_stranger_pets.py --apply --threshold 0.15
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

from dotenv import load_dotenv

# Silence model-loading chatter from imports below.
for noisy in ("insightface", "onnxruntime", "ultralytics"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from ritrova.cluster import cluster_faces  # noqa: E402
from ritrova.db import FaceDB  # noqa: E402

PET_SPECIES = ("dog", "cat")
SPECIES_TO_LABEL = {"dog": "Dog", "cat": "Cat"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    parser.add_argument("--apply", action="store_true", help="Execute the migration.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Cosine-distance threshold for re-clustering (smaller = tighter). Default 0.18.",
    )
    parser.add_argument("--min-size", type=int, default=2, help="Minimum cluster size.")
    parser.add_argument(
        "--dissolve-name",
        action="append",
        default=None,
        help="Subject name (case-insensitive) to dissolve. Repeatable. "
        "Default: 'Cani a caso', 'Gatti a caso'.",
    )
    args = parser.parse_args()
    if args.dry_run == args.apply:
        raise SystemExit("Pass exactly one of --dry-run / --apply.")

    dissolve_names = args.dissolve_name or ["Cani a caso", "Gatti a caso"]

    load_dotenv()
    db_path = os.environ.get("FACE_DB", "./data/faces.db")
    db = FaceDB(db_path)

    # ── Step 1: identify dumpster subjects ────────────────────────────
    placeholders = ",".join("?" * len(dissolve_names))
    rows = db.conn.execute(
        f"SELECT id, name FROM subjects WHERE kind='pet' AND LOWER(name) IN ({placeholders})",
        tuple(n.lower() for n in dissolve_names),
    ).fetchall()
    dumpsters: list[tuple[int, str]] = [(int(r[0]), r[1]) for r in rows]
    if not dumpsters:
        print(f"No dumpster subjects match {dissolve_names}. Nothing to do.")
        return

    total_to_free = 0
    for sid, name in dumpsters:
        n = db.conn.execute("SELECT COUNT(*) FROM findings WHERE person_id = ?", (sid,)).fetchone()[
            0
        ]
        print(f"  Dumpster: '{name}' (id={sid}) — {n} findings to unassign")
        total_to_free += n
    print(f"Total findings to free: {total_to_free}")

    # ── Step 2/3 preview: plus existing unassigned + current unnamed clusters ─
    unassigned_by_species = dict(
        db.conn.execute(
            "SELECT species, COUNT(*) FROM findings "
            "WHERE person_id IS NULL AND species IN ('dog','cat') "
            "GROUP BY species"
        ).fetchall()
    )
    print(f"Existing unassigned pet findings (pre-migration): {unassigned_by_species}")

    # ── Step 4/5: preview by running a dry-run clustering count. ──────
    # We don't actually re-cluster in dry-run — just estimate.
    print(
        f"\nPlanned re-cluster: threshold={args.threshold}, "
        f"min_size={args.min_size}, species={PET_SPECIES}"
    )
    print(
        "After migration, every resulting unnamed cluster will become a "
        "`StrangerDog #N` / `StrangerCat #N` subject enrolled in the Strangers circle."
    )

    if args.dry_run:
        print("\nDry run — no changes. Run with --apply to execute.")
        return

    # ── Step 2 + 3: dissolve dumpsters ─────────────────────────────────
    dumpster_ids = [sid for sid, _ in dumpsters]
    db.conn.execute("BEGIN")
    db.conn.executemany(
        "UPDATE findings SET person_id = NULL WHERE person_id = ?",
        [(sid,) for sid in dumpster_ids],
    )
    db.conn.executemany(
        "DELETE FROM subjects WHERE id = ?",
        [(sid,) for sid in dumpster_ids],
    )
    db.conn.commit()
    print(f"\nDissolved {len(dumpster_ids)} dumpster subject(s).")

    # ── Step 4: re-cluster each pet species. ──────────────────────────
    for species in PET_SPECIES:
        print(f"\n── clustering {species} (threshold={args.threshold}) ──")
        result = cluster_faces(
            db, threshold=args.threshold, min_size=args.min_size, species=species
        )
        print(
            f"  faces={result['total_faces']}, clusters={result['clusters']}, "
            f"noise={result['noise']}, largest={result.get('largest_cluster', 0)}"
        )

    # ── Step 5: mint StrangerDog/Cat #N for every unnamed pet cluster. ─
    strangers = db.get_circle_by_name("Strangers")
    if strangers is None:
        raise SystemExit("Strangers circle missing — unexpected. DB schema out of date?")

    minted_by_species: dict[str, int] = defaultdict(int)
    # Pick majority species per unnamed pet cluster. A cluster can in principle
    # hold mixed species at loose thresholds; at 0.18 it's almost always pure.
    rows = db.conn.execute(
        """
        SELECT cluster_id, species, COUNT(*) AS n
        FROM findings
        WHERE cluster_id IS NOT NULL AND person_id IS NULL
        AND species IN ('dog','cat')
        GROUP BY cluster_id, species
        """
    ).fetchall()
    by_cluster: dict[int, tuple[str, int]] = {}
    for cluster_id, sp, n in rows:
        prev = by_cluster.get(cluster_id)
        if prev is None or n > prev[1]:
            by_cluster[cluster_id] = (sp, n)
    per_cluster_species = [(cid, sp) for cid, (sp, _) in by_cluster.items()]

    existing_n_per_label: dict[str, int] = {}
    for species in PET_SPECIES:
        label = f"Stranger{SPECIES_TO_LABEL[species]}"
        prefix_len = len(label) + 2  # "Label #" → +2 = "Label #" prefix length
        row = db.conn.execute(
            f"SELECT COALESCE(MAX(CAST(SUBSTR(name, {prefix_len}) AS INTEGER)), 0) "
            f"FROM subjects WHERE kind='pet' AND name LIKE ?",
            (f"{label} #%",),
        ).fetchone()
        existing_n_per_label[label] = int(row[0] or 0)

    for cluster_id, species in per_cluster_species:
        if species not in PET_SPECIES:
            continue
        label = f"Stranger{SPECIES_TO_LABEL[species]}"
        existing_n_per_label[label] += 1
        name = f"{label} #{existing_n_per_label[label]}"
        subject_id = db.create_subject(name, kind="pet")
        db.assign_cluster_to_subject(cluster_id, subject_id)
        db.add_subject_to_circle(subject_id, strangers.id)
        minted_by_species[species] += 1

    print(f"\nMinted per-cluster stranger subjects: {dict(minted_by_species)}")

    # Final summary
    final = dict(
        db.conn.execute(
            "SELECT species, COUNT(*) FROM findings "
            "WHERE person_id IS NULL AND species IN ('dog','cat') "
            "GROUP BY species"
        ).fetchall()
    )
    print(f"Remaining unassigned pet findings (singletons + below min_size): {final}")


if __name__ == "__main__":
    main()
