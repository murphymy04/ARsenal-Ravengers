"""Flask companion web app for managing people and viewing interactions.

Provides a dashboard for:
- Viewing and editing enrolled people
- Enrolling new people from uploaded images
- Viewing interaction history and transcripts
- Speaker-to-person mapping

Stubbed for Phase 2 implementation.
"""



class CompanionApp:
    """Flask-based companion web application (stub - Phase 2)."""

    def __init__(self, db):
        self.db = db
        self._app = None

    def create_app(self):
        """Create and configure the Flask application."""
        raise NotImplementedError("Companion app is Phase 2 - requires Flask")

    def run(self):
        """Run the Flask development server (blocking)."""
        raise NotImplementedError("Companion app is Phase 2 - requires Flask")

    def run_threaded(self):
        """Run the Flask server in a daemon thread."""
        raise NotImplementedError("Companion app is Phase 2 - requires Flask")
