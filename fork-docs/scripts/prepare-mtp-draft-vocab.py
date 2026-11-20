"""Generate reproducible multilingual numeric IDs solely from the model's GGUF tokenizer metadata."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ALGORITHM = "gguf-script-quota-v1"


DIRECT_BYTES = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
MISSING_BYTES = [value for value in range(256) if value not in DIRECT_BYTES]
BYTE_INVERSE = dict(zip(map(chr, DIRECT_BYTES + list(range(256, 256 + len(MISSING_BYTES)))), DIRECT_BYTES + MISSING_BYTES))


def token_category(piece):
    inverse = BYTE_INVERSE
    if all(char in inverse for char in piece):
        text = bytes(inverse[char] for char in piece).decode("utf-8", errors="replace")
    else:
        text = piece
    points = [ord(char) for char in text]
    if any(0x3040 <= char <= 0x30FF for char in points):
        return "kana"
    if any(0x1100 <= char <= 0x11FF or 0xAC00 <= char <= 0xD7AF for char in points):
        return "hangul"
    if any(0x3400 <= char <= 0x9FFF or 0x20000 <= char <= 0x323AF for char in points):
        return "han"
    if text and all(ord(char) < 128 for char in text):
        return "ascii"
    return "other"


def read_tokenizer(path, library=Path("G:/AI/build-prefill32/bin/ggml-base.dll")):
    import ctypes
    import os
    from contextlib import nullcontext

    class Init(ctypes.Structure):
        _fields_ = [("no_alloc", ctypes.c_bool), ("ctx", ctypes.c_void_p)]

    with os.add_dll_directory(str(library.resolve().parent)) if os.name == "nt" else nullcontext():
        lib = ctypes.CDLL(str(library.resolve()))
    signatures = {
        "gguf_init_from_file": ([ctypes.c_char_p, Init], ctypes.c_void_p),
        "gguf_free": ([ctypes.c_void_p], None),
        "gguf_find_key": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int64),
        "gguf_get_n_kv": ([ctypes.c_void_p], ctypes.c_int64),
        "gguf_get_key": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_char_p),
        "gguf_get_kv_type": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
        "gguf_get_arr_type": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
        "gguf_get_arr_n": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_size_t),
        "gguf_get_arr_str": ([ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t], ctypes.c_char_p),
        "gguf_get_arr_data": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_void_p),
        "gguf_get_val_u32": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_uint32),
        "gguf_get_val_str": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_char_p),
    }
    for name, (arguments, result) in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = result
    ctx = lib.gguf_init_from_file(str(path.resolve()).encode("utf-8"), Init(True, None))
    if not ctx:
        raise ValueError("GGUF metadata parsing failed")
    try:
        token_key = lib.gguf_find_key(ctx, b"tokenizer.ggml.tokens")
        type_key = lib.gguf_find_key(ctx, b"tokenizer.ggml.token_type")
        if min(token_key, type_key) < 0:
            raise ValueError("missing tokenizer arrays")
        if (lib.gguf_get_kv_type(ctx, token_key) != 9 or lib.gguf_get_arr_type(ctx, token_key) != 8
                or lib.gguf_get_kv_type(ctx, type_key) != 9 or lib.gguf_get_arr_type(ctx, type_key) != 5):
            raise ValueError("unexpected tokenizer metadata types")
        count = lib.gguf_get_arr_n(ctx, token_key)
        if not 16 < count <= 1024 * 1024 or lib.gguf_get_arr_n(ctx, type_key) != count:
            raise ValueError("unsupported tokenizer array length")
        tokens = [lib.gguf_get_arr_str(ctx, token_key, i).decode("utf-8") for i in range(count)]
        values = ctypes.cast(lib.gguf_get_arr_data(ctx, type_key), ctypes.POINTER(ctypes.c_int32))
        types = [int(values[i]) for i in range(count)]
        special, provenance = {}, {}
        for i in range(lib.gguf_get_n_kv(ctx)):
            name = lib.gguf_get_key(ctx, i).decode("utf-8")
            kind = lib.gguf_get_kv_type(ctx, i)
            if name.startswith("tokenizer.") and name.endswith("_token_id") and kind == 4:
                special[name] = int(lib.gguf_get_val_u32(ctx, i))
            if name in ("general.name", "general.license", "general.license.link", "general.source.url") and kind == 8:
                provenance[name] = lib.gguf_get_val_str(ctx, i).decode("utf-8")
        digest = hashlib.sha256()
        for token, kind in zip(tokens, types):
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little") + encoded + kind.to_bytes(4, "little"))
        return tokens, types, digest.hexdigest(), special, provenance
    finally:
        lib.gguf_free(ctx)


def select_ids(tokens, types, size, special):
    if not 16 <= size < len(tokens):
        raise ValueError("size must be16..vocab-1")
    categories = {name: [] for name in ("ascii", "han", "kana", "hangul", "other")}
    mandatory = {i for i, kind in enumerate(types) if kind != 1}
    for value in special.values():
        if isinstance(value, int) and 0 <= value < len(tokens):
            mandatory.add(value)
    if len(mandatory) > size:
        raise ValueError("requested size cannot contain all specials/byte tokens")
    for i, token in enumerate(tokens):
        categories[token_category(token)].append(i)
    selected = set(mandatory)
    # Equal algorithm and vocabulary order give identical IDs, without a mutable text corpus.
    for name, fraction in (("ascii", 0.50), ("han", 0.25), ("kana", 0.125), ("hangul", 0.0625), ("other", 0.0625)):
        count = 0
        for token in categories[name]:
            if token not in selected and len(selected) < size and count < int(size * fraction):
                selected.add(token)
                count += 1
    for token in range(len(tokens)):
        if len(selected) == size:
            break
        selected.add(token)
    if len(selected) != size or not mandatory <= selected:
        raise ValueError("subset construction failed")
    counts = {name: sum(token in selected for token in ids) for name, ids in categories.items()}
    return sorted(selected), counts, len(mandatory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True, help="target or draft GGUF containing tokenizer metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ggml-base", type=Path, default=Path("G:/AI/build-prefill32/bin/ggml-base.dll"), help="metadata parser DLL; no model/tensor allocation")
    parser.add_argument("--size", type=int, default=65536)
    args = parser.parse_args()
    tokens, types, tokenizer_sha, special, provenance = read_tokenizer(args.gguf, args.ggml_base)
    ids, counts, mandatory = select_ids(tokens, types, args.size, special)
    payload = ("# " + ALGORITHM + "\n" + "\n".join(map(str, ids)) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {"algorithm": ALGORITHM, "source_gguf": str(args.gguf.resolve()), "source": "model tokenizer metadata only; no external or local text corpus",
                "tokenizer_sha256": tokenizer_sha, "list_sha256": hashlib.sha256(payload).hexdigest(),
                "vocab_size": len(tokens), "subset_size": len(ids), "mandatory_tokens": mandatory,
                "selected_scripts": counts, "special_token_fields": special, "model_license_metadata": provenance}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
