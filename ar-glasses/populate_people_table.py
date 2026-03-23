import random
from storage.database import Database

NAMES = [
    "Alex Johnson",
    "Taylor Perez",
    "Jordan Smith",
    "Riley Williams",
    "Morgan Davis",
    "Casey Brown",
    "Jamie Lee",
    "Drew Miller",
    "Parker Wilson",
    "Skylar Moore",
]

NOTES = [
    "Test user",
    "Needs review",
    "No notes",
    "Early candidate",
    "Frequent visitor",
]


def populate(n: int = 10):
    db = Database()
    print(f"Using DB: {db._conn}")

    added = []
    for i in range(n):
        name = random.choice(NAMES)
        note = random.choice(NOTES)
        # Use explicit is_labeled False to mimic newly discovered person
        pid = db.add_person(name=name, notes=note, is_labeled=False)
        added.append((pid, name, note))

    print(f"Added {len(added)} people:")
    for pid, name, note in added:
        print(f"  - {pid}: {name} ({note})")

    total = db._conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    unlabeled = db._conn.execute("SELECT COUNT(*) FROM people WHERE is_labeled = 0").fetchone()[0]
    print(f"Total people in DB: {total}")
    print(f"Unlabeled people in DB: {unlabeled}")


if __name__ == "__main__":
    populate(10)
