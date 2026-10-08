import pytest

from app.history import history_store


@pytest.fixture(autouse=True)
def _no_history_database():
    """Tests never write to the real AGENT_DB_NAME database; tests that need history
    point the store at a throwaway database (see test_contact_us_history.py)."""
    history_store.configure(enabled=False)
    yield
    history_store.configure(enabled=False)
