# Telugu symbol set (replaces the English/ARPAbet set of the reference implementation)
_pad = "_"
_special = "-"
_punctuation = "!'(),.:;? "
_telugu = [chr(c) for c in range(0x0C00, 0x0C80)]        # Telugu Unicode block (graphemes = MFA "phones")
_digits = list("0123456789")                              # digits only so unseen input never crashes
_arpabet = ["@" + s for s in _telugu + _digits]
_silences = ["@sp", "@spn", "@sil"]
symbols = [_pad] + list(_special) + list(_punctuation) + _arpabet + _silences
