"""Shared strict tokenizer utilities for the Hindi VITS experiment.

The experiment uses space-delimited phoneme/cluster tokens, not characters.
Use these helpers in both training and inference so multi-character tokens such
as ``aa``, ``bh``, and ``C10`` retain exactly the same representation.
"""

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<blnk>", "<wb>")

BASELINE_PHONEMES = (
    "a", "aa", "ae", "ax", "b", "bh", "c", "ch", "d", "dh",
    "dx", "dxh", "dxhq", "dxq", "ee", "ei", "f", "g", "gh", "gq",
    "h", "hq", "i", "ii", "j", "jh", "k", "kh", "khq", "kq",
    "l", "lx", "m", "mq", "n", "ng", "nj", "nx", "o", "ou",
    "p", "ph", "q", "r", "rq", "s", "sh", "sx", "t", "th",
    "tx", "txh", "u", "uu", "w", "y", "z",
)

CLUSTER_TOKENS = tuple(f"C{i}" for i in range(39))


def build_vocab(content_tokens):
    """Return the fixed experiment token-to-ID mapping."""
    tokens = list(SPECIAL_TOKENS) + list(content_tokens)
    if len(tokens) != len(set(tokens)):
        raise ValueError("Tokenizer vocabulary contains duplicate tokens")
    return {token: index for index, token in enumerate(tokens)}


def validate_token_sequence(text, vocab):
    """Validate a whitespace-delimited sequence and return its tokens.

    Unknown tokens are an error. Silently dropping them would corrupt the
    linguistic input and invalidate a baseline-versus-clustered comparison.
    """
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("Empty token sequence")
    unknown = [token for token in tokens if token not in vocab]
    if unknown:
        raise ValueError(f"Unknown token(s): {unknown}")
    return tokens


def patch_tokenizer(tokenizer, vocab):
    """Configure a Coqui tokenizer to use strict whitespace tokenization."""
    id_to_token = {index: token for token, index in vocab.items()}
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    blank_id = vocab["<blnk>"]

    def text_to_ids(text):
        return [vocab[token] for token in validate_token_sequence(text, vocab)]

    def ids_to_text(ids):
        special_ids = {pad_id, bos_id, eos_id, blank_id}
        return " ".join(
            id_to_token[token_id]
            for token_id in ids
            if token_id in id_to_token and token_id not in special_ids
        )

    tokenizer.text_to_ids = text_to_ids
    tokenizer.ids_to_text = ids_to_text
    tokenizer.vocab_size = len(vocab)
    tokenizer.pad_id = pad_id
    tokenizer.blank_id = blank_id
    tokenizer.bos_id = bos_id
    tokenizer.eos_id = eos_id

    # Keep Coqui's character container consistent with the patched tokenizer.
    # VITS consults this object for vocabulary-size and special-token metadata.
    chars = getattr(tokenizer, "characters", None)
    if chars is not None:
        chars._char_to_id = vocab
        chars._id_to_char = id_to_token
        chars.char_to_id = lambda token: vocab[token]
        chars.id_to_char = lambda token_id: id_to_token[token_id]
        chars.vocab_size = len(vocab)
        chars.num_chars = len(vocab)
        chars.pad_id = pad_id
        chars.blank_id = blank_id
        chars.pad = "<pad>"
        chars.blank = "<blnk>"

    return tokenizer


def assert_tokenizer_round_trip(tokenizer, sequence):
    """Raise if a tokenizer cannot faithfully encode and decode a sequence."""
    ids = tokenizer.text_to_ids(sequence)
    decoded = tokenizer.ids_to_text(ids)
    if decoded != sequence:
        raise AssertionError(
            f"Tokenizer round trip failed: {sequence!r} -> {ids} -> {decoded!r}"
        )
    return ids
