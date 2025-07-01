"""Basic import test to ensure CI has at least one passing test."""

def test_can_import_music_gen():
    """Test that music_gen package can be imported."""
    import music_gen
    assert music_gen.__version__ == "0.1.0"


def test_can_import_exceptions():
    """Test that exceptions can be imported."""
    from music_gen.utils.exceptions import MusicGenException
    assert MusicGenException is not None


def test_can_import_logging():
    """Test that logging can be imported.""" 
    from music_gen.utils.logging import get_logger
    logger = get_logger("test")
    assert logger is not None
