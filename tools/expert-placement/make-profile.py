"""Create a Flash-Next resident Expert profile from public activation-map hits."""

import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activation_map", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    hits = [[0] * 512 for _ in range(48)]
    layers = {str(i) for i in range(48)}
    count = 0
    with gzip.open(args.activation_map, "rt", encoding="utf-8") as source:
        for row in source:
            counts = json.loads(row)["layers"]
            if counts.keys() != layers:
                raise ValueError(f"Row {count + 1}: expected layers 0 through 47")
            for layer, stats in counts.items():
                values = stats["hits"]
                if len(values) != 512 or any(type(v) is not int or v < 0 for v in values):
                    raise ValueError(f"Row {count + 1}, layer {layer}: expected 512 hit counts")
                hits[int(layer)] = [a + b for a, b in zip(hits[int(layer)], values)]
            count += 1

    if not count or any(sum(layer) == 0 for layer in hits):
        raise ValueError("The activation map has no routing observations for one or more layers")

    with args.output.open("w", encoding="ascii", newline="\n") as output:
        for layer, counts in enumerate(hits):
            ranked = sorted(range(512), key=lambda expert: (-counts[expert], expert))[:300]
            output.write(f"{layer} " + " ".join(map(str, sorted(ranked))) + "\n")


if __name__ == "__main__":
    main()
