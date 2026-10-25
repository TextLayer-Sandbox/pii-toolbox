"""Co-reference worker: answers JSON requests on a ZeroMQ ROUTER socket with mention clusters.

A request is {"id": int, "text": str, "names": [[start, end], ...]} and the reply is
{"id": int, "clusters": [[[start, end], ...], ...]} with character positions only, so no
document text ever leaves this process. Runs in its own uv environment; `--stub` swaps the
model for a pattern-based clusterer so the protocol can be exercised without model files.
"""

import argparse
import atexit
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import zmq

POLL_INTERVAL_MS = 200
MAVERICK_DIRECTORY = "sapienzanlp__maverick-mes-ontonotes"
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_MODEL_DIRECTORY = "en_core_web_sm"
PUNKT_DIRECTORY = "nltk_punkt"
_NAME_RUN = re.compile(r"(?:(?:Mr|Mrs|Ms|Dr|Prof)\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")


def stub_clusters(text: str) -> list[list[tuple[int, int]]]:
    """Clusters capitalised name runs that share a last word, reporting only clusters of two or more mentions."""
    by_last_word: dict[str, list[tuple[int, int]]] = {}
    for match in _NAME_RUN.finditer(text):
        by_last_word.setdefault(match.group().split()[-1], []).append((match.start(), match.end()))
    return [cluster for cluster in by_last_word.values() if len(cluster) > 1]


def load_offline_text_resources(model_dir: Path) -> None:
    """Points Maverick's tokeniser lookups at files under the model directory so nothing is downloaded."""
    import nltk
    import spacy
    from maverick.models import maverick_model

    scratch = Path(tempfile.mkdtemp(prefix="coref-resources-"))
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    (spacy_wheel,) = (model_dir / SPACY_MODEL_DIRECTORY).glob("*.whl")
    with zipfile.ZipFile(spacy_wheel) as wheel:
        wheel.extractall(scratch / "spacy")
    (spacy_package,) = (scratch / "spacy" / SPACY_MODEL_NAME).glob(f"{SPACY_MODEL_NAME}-*")
    with zipfile.ZipFile(model_dir / PUNKT_DIRECTORY / "punkt.zip") as punkt:
        punkt.extractall(scratch / "nltk" / "tokenizers")
    nltk.data.path.insert(0, str(scratch / "nltk"))
    nlp = spacy.load(spacy_package, exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
    maverick_model.download_load_spacy = lambda: nlp


def maverick_clusters_factory(model_dir: Path):
    import collections
    import typing

    import omegaconf
    import torch
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    # The checkpoint pickles these configuration types; allow-listing them keeps weights_only loading on.
    torch.serialization.add_safe_globals(
        [omegaconf.DictConfig, omegaconf.ListConfig, ContainerMetadata, Metadata, AnyNode, typing.Any, dict, list, collections.defaultdict]
    )
    from maverick import Maverick

    load_offline_text_resources(model_dir)
    model = Maverick(hf_name_or_path=str(model_dir / MAVERICK_DIRECTORY), device="cpu")

    def predict(text: str) -> list[list[tuple[int, int]]]:
        output = model.predict(text)
        # The model reports the offset of each mention's last character; end positions here are exclusive.
        return [[(start, end + 1) for start, end in cluster] for cluster in output["clusters_char_offsets"]]

    return predict


def overlaps_any(mention: tuple[int, int], names: list[list[int]]) -> bool:
    return any(mention[0] < end and start < mention[1] for start, end in names)


def handle(raw: bytes, predict) -> bytes:
    request = json.loads(raw)
    clusters = predict(request["text"])
    kept = [cluster for cluster in clusters if any(overlaps_any(m, request["names"]) for m in cluster)]
    return json.dumps({"id": request["id"], "clusters": kept}).encode()


def serve(endpoint: str, predict) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(endpoint)
    try:
        while True:
            if not socket.poll(POLL_INTERVAL_MS):
                continue
            identity, raw = socket.recv_multipart()
            socket.send_multipart([identity, handle(raw, predict)])
    finally:
        socket.close(linger=0)
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="ipc:// endpoint to bind")
    parser.add_argument("--model-dir", type=Path, help="model directory holding the Maverick, spaCy and punkt files")
    parser.add_argument("--stub", action="store_true", help="use the pattern-based clusterer")
    args = parser.parse_args()
    if args.stub:
        predict = stub_clusters
    elif args.model_dir:
        predict = maverick_clusters_factory(args.model_dir)
    else:
        parser.error("either --model-dir or --stub is required")
    serve(args.endpoint, predict)


if __name__ == "__main__":
    main()
