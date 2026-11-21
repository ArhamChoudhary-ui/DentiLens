"""Install the public Whisper tiny weights for offline inference."""
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
snapshot_download('openai/whisper-tiny', revision='169d4a4341b33bc18d8881c4b69c2e104e1cc0af',
                  allow_patterns=['*.json', '*.txt', 'model.safetensors'], local_dir=root / 'models/whisper-tiny')
