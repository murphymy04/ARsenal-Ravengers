"""CLI commands for inspecting and managing the face database."""

from storage.database import Database


def db_info_mode(db: Database):
    """Print a summary of everyone in the database."""
    people = db.get_all_people()
    if not people:
        print("Database is empty — no people enrolled yet.")
        return

    print(f"\n{'─' * 55}")
    print(f"  {'ID':<5} {'Name':<20} {'Embeddings':<12} {'Last Seen'}")
    print(f"{'─' * 55}")
    for p in people:
        last_seen = p.last_seen.strftime("%Y-%m-%d %H:%M") if p.last_seen else "never"
        print(f"  {p.person_id:<5} {p.name:<20} {len(p.embeddings):<12} {last_seen}")
    print(f"{'─' * 55}")
    print(f"  Total: {len(people)} person(s)\n")


def db_delete_mode(db: Database):
    """Interactively delete a person or wipe the entire database."""
    people = db.get_all_people()
    if not people:
        print("Database is empty.")
        return

    db_info_mode(db)
    print("Options:")
    print("  Enter a person ID to delete that person")
    print("  Enter 'all' to wipe the entire database")
    print("  Enter 'cancel' to exit")

    choice = input("\nChoice: ").strip().lower()

    if choice == "cancel":
        print("Cancelled.")
        return

    if choice == "all":
        confirm = input(
            f"Delete ALL {len(people)} people? This cannot be undone. (yes/no): "
        ).strip().lower()
        if confirm == "yes":
            for p in people:
                db.delete_person(p.person_id)
            print("Database wiped.")
        else:
            print("Cancelled.")
        return

    try:
        person_id = int(choice)
    except ValueError:
        print(f"Invalid input: '{choice}'")
        return

    person = db.get_person(person_id)
    if not person:
        print(f"No person found with ID {person_id}.")
        return

    confirm = input(
        f"Delete '{person.name}' (ID {person_id})? (yes/no): "
    ).strip().lower()
    if confirm == "yes":
        db.delete_person(person_id)
        print(f"Deleted '{person.name}'.")
    else:
        print("Cancelled.")
