"""pytest configuration: add project root to Python path so tests can import src.*"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
