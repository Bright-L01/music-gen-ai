"""
Training script with Hydra configuration support.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from music_gen.training.hydra_trainer import train

if __name__ == "__main__":
    train()