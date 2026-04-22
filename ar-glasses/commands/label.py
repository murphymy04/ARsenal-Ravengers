"""CLI command for assigning names to auto-discovered face clusters."""

import cv2
import numpy as np

from storage.database import Database
from commands.merge import do_merge


def label_mode(db: Database):
    """Interactively assign real names to auto-discovered clusters."""
    people = db.get_all_people()
    unlabeled = [p for p in people if not p.is_labeled]

    if not unlabeled:
        print("No unlabeled clusters found. Run in 'run' mode first to discover faces.")
        return

    print(f"\nFound {len(unlabeled)} unlabeled cluster(s).")
    print("Commands: type a name to label, press Enter to skip, 'delete' to remove.\n")

    for person in unlabeled:
        last_seen = (
            person.last_seen.strftime("%Y-%m-%d %H:%M") if person.last_seen else "never"
        )
        print(
            f"Cluster '{person.name}' | {len(person.embeddings)} embedding(s) | last seen: {last_seen}"
        )
        print("  Type name + Enter | Esc to skip | type 'delete' + Enter to remove")

        if person.thumbnail is not None:
            name = _cv2_input(person.thumbnail, "Name")
        else:
            name = input("  Name: ").strip()

        if name.lower() == "delete":
            db.delete_person(person.person_id)
            print(f"  Deleted '{person.name}'.")
        elif name:
            existing = db.get_person_by_name(name)
            if existing:
                do_merge(db, keep=existing, discard=person)
                print(f"  Merged into existing '{name}'")
            else:
                db.update_person(person.person_id, name=name, is_labeled=True)
                print(f"  '{person.name}' → '{name}'")
        else:
            print("  Skipped.")

    print("\nLabeling complete.")


def _cv2_input(image: np.ndarray, prompt: str) -> str:
    """Show an image in a cv2 window and collect keyboard input.

    Returns the typed string on Enter, or empty string on Esc.
    Supports Backspace for corrections.
    """
    typed = ""
    win = "Labeling"
    while True:
        display = image.copy()
        cv2.putText(
            display,
            f"{prompt}: {typed}_",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter — confirm
            break
        elif key == 27:  # Esc — skip
            typed = ""
            break
        elif key in (8, 127):  # Backspace
            typed = typed[:-1]
        elif 32 <= key < 127:  # Printable ASCII
            typed += chr(key)

    cv2.destroyWindow(win)
    return typed
