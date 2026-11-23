"""Stage 1: a character-level tokenizer.

A neural net only understands numbers, so the tokenizer is the bridge:
    text  --encode-->  list[int]  --model-->  list[int]  --decode-->  text
"""


def build_vocab(text):
    """Map every distinct character in `text` to an integer id, and back.

    sorted() matters: set() has no guaranteed order, so without sorting the
    same text could produce different ids on different runs. A trained model
    is tied to its ids, so the mapping must be deterministic.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int (for encode)
    itos = {i: ch for i, ch in enumerate(chars)}  # int -> string (for decode)
    return stoi, itos


def encode(s, stoi):
    return [stoi[ch] for ch in s]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


if __name__ == "__main__":
    # Tiny self-check: encoding then decoding must give back the original text.
    with open("input.txt", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    print(f"characters in file: {len(text):,}")
    print(f"vocab size:         {len(stoi)}")
    print("vocab:", "".join(itos[i] for i in range(len(itos))).__repr__())

    sample = "To be, or not to be"
    ids = encode(sample, stoi)
    print(f"{sample!r} -> {ids}")
    assert decode(ids, itos) == sample
    assert decode(encode(text, stoi), itos) == text
    print("round trip OK")
