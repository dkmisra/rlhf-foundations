from rlhf2.tasks.dyck import DyckTokenizer


def test_tokenize_decode_round_trip():
    tok = DyckTokenizer()
    prompt = "([{}])"

    ids = [tok.token_to_idx[t] for t in tok.tokenize(prompt)]
    assert tok.decode(ids) == prompt


def test_batch_encode_pads_to_longest_and_masks():
    tok = DyckTokenizer()
    input_ids, attention_mask = tok.batch_encode(["(", "([{"], padding=True)

    # Both rows padded to the longest sequence (length 3).
    assert [len(row) for row in input_ids] == [3, 3]
    assert input_ids[0][1:] == [tok.PAD_ID, tok.PAD_ID]
    assert attention_mask[0] == [1.0, 0.0, 0.0]
    assert attention_mask[1] == [1.0, 1.0, 1.0]


def test_decode_stops_at_eos_and_skips_pad():
    tok = DyckTokenizer()
    # pad, "(", eos, ")"  -> pad skipped, decode stops at eos.
    ids = [tok.PAD_ID, tok.token_to_idx["("], tok.eos, tok.token_to_idx[")"]]
    assert tok.decode(ids) == "("
