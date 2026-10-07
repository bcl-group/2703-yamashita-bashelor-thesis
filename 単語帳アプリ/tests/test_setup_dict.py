from setup_dict import concat_ejdict


def test_concat_ejdict_drops_blank_and_malformed():
    out = concat_ejdict(["a\tあ\n\n", "b\tび\nbroken line\n"])
    assert out.splitlines() == ["a\tあ", "b\tび"]
