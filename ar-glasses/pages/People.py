"""People management page — label and delete faces from the database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import streamlit as st

from storage.database import Database
from config import MIN_SIGHTINGS_TO_CLUSTER

st.set_page_config(page_title="ARsenal — People", layout="wide", page_icon="👥")
st.title("👥 People Database")

db = Database()

# Handle pending actions
if "_pending_rename" in st.session_state:
    pid, new_name = st.session_state.pop("_pending_rename")
    db.update_person(pid, name=new_name, is_labeled=True)
    st.toast(f"Renamed to '{new_name}'", icon="✅")

if "_pending_delete" in st.session_state:
    pid = st.session_state.pop("_pending_delete")
    db.delete_person(pid)
    st.toast("Person deleted", icon="🗑️")

people = db.get_all_people()
db.close()

if not people:
    st.info(
        f"No people in the database yet. "
        f"Faces auto-cluster after being seen {MIN_SIGHTINGS_TO_CLUSTER} times."
    )
else:
    st.caption(f"{len(people)} people in database")
    COLS = 4
    for row_start in range(0, len(people), COLS):
        row = people[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col, person in zip(cols, row):
            with col:
                if person.thumbnail is not None:
                    st.image(
                        cv2.cvtColor(person.thumbnail, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                else:
                    st.write("📷 No image")

                badge = "✅" if person.is_labeled else "❓ auto"
                last_seen = (
                    person.last_seen.strftime("%H:%M") if person.last_seen else "—"
                )
                st.caption(
                    f"{badge} · {len(person.embeddings)} embeddings · last seen {last_seen}"
                )

                new_name = st.text_input(
                    "Name",
                    value=person.name,
                    key=f"name_{person.person_id}",
                    label_visibility="collapsed",
                    placeholder="Enter name…",
                )
                btn_save, btn_del = st.columns(2)
                with btn_save:
                    if st.button(
                        "💾 Save",
                        key=f"save_{person.person_id}",
                        use_container_width=True,
                    ):
                        st.session_state["_pending_rename"] = (
                            person.person_id,
                            new_name,
                        )
                        st.rerun()
                with btn_del:
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_{person.person_id}",
                        use_container_width=True,
                    ):
                        st.session_state["_pending_delete"] = person.person_id
                        st.rerun()

if st.button("🔄 Refresh"):
    st.rerun()
