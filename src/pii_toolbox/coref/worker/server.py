"""Co-reference worker: answers JSON requests on a ZeroMQ ROUTER socket with mention clusters.

A request is {"id": int, "text": str, "names": [[start, end], ...]} and the reply is
{"id": int, "clusters": [[[start, end], ...], ...]} with character positions only, so no
document text ever leaves this process. Runs in its own uv environment; `--stub` swaps the
model for a pattern-based clusterer so the protocol can be exercised without model files.
"""

import argparse
import json
import re

import zmq

POLL_INTERVAL_MS = 200
_NAME_RUN = re.compile(r"(?:(?:Mr|Mrs|Ms|Dr|Prof)\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")


def stub_clusters(text: str) -> list[list[tuple[int, int]]]:
    """Clusters every capitalised name run by its last word; a deterministic stand-in for the model."""
    by_last_word: dict[str, list[tuple[int, int]]] = {}
    for match in _NAME_RUN.finditer(text):
        by_last_word.setdefault(match.group().split()[-1], []).append((match.start(), match.end()))
    return list(by_last_word.values())


def maverick_clusters_factory(model_path: str):
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

    model = Maverick(hf_name_or_path=model_path, device="cpu")

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
    parser.add_argument("--model", help="directory holding the Maverick model files")
    parser.add_argument("--stub", action="store_true", help="use the pattern-based clusterer")
    args = parser.parse_args()
    if args.stub:
        predict = stub_clusters
    elif args.model:
        predict = maverick_clusters_factory(args.model)
    else:
        parser.error("either --model or --stub is required")
    serve(args.endpoint, predict)


if __name__ == "__main__":
    main()
