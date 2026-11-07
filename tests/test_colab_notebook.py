import json
from pathlib import Path


def test_colab_training_happens_before_adapter_download() -> None:
    notebook = json.loads(Path("notebooks/train_colab.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    upload_index = next(index for index, source in enumerate(sources) if "files.upload()" in source)
    training_index = next(index for index, source in enumerate(sources) if "!toolforge-train" in source)
    download_index = next(index for index, source in enumerate(sources) if "files.download" in source)

    assert upload_index < training_index < download_index

