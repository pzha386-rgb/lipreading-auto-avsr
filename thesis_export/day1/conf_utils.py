"""
conf_utils.py  -  Day 1 shared tools
- normalize(): same idea as before (lowercase, numbers -> words)
- wer_counts(): word errors + reference length (for corpus WER)
- decode_nbest_autoavsr(): same path as ModelModule.forward(),
  but keeps top-k hypotheses and their scores
- nbest_features(): turns N-best into confidence scores (method A)
"""
import re
import math

try:
    import inflect
    _P = inflect.engine()
except ImportError:
    _P = None


def normalize(text):
    t = str(text).lower()
    if _P is not None:
        t = re.sub(r"\d+(\.\d+)?", lambda m: _P.number_to_words(m.group()), t)
    t = t.replace("-", " ").replace("\u2019", " ").replace("'", " ")   # we're -> we re (same as benchmark)
    t = re.sub(r"[^a-z ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer_counts(ref, hyp):
    """returns (errors, ref_len) after normalization"""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    return edit_distance(r, h), len(r)


def text_disagreement(a, b):
    """symmetric word-level difference between two hypotheses, 0..1"""
    x, y = normalize(a).split(), normalize(b).split()
    if not x and not y:
        return 0.0
    return min(1.0, edit_distance(x, y) / max(len(x), len(y)))


# ---------------------------------------------------------------- Auto-AVSR
def decode_nbest_autoavsr(module, sample, k=5):
    """
    Drop-in replacement for:   pred = module(sample)
    Use:                       nbest = decode_nbest_autoavsr(module, sample)
                               pred  = nbest[0]["text"]
    `module` = your loaded Auto-AVSR ModelModule (eval mode)
    `sample` = the same tensor you already pass to module(sample)
    """
    import torch
    from lightning import get_beam_search_decoder  # from the auto_avsr repo

    with torch.no_grad():
        if getattr(module, "_nbest_bs", None) is None:
            module._nbest_bs = get_beam_search_decoder(module.model, module.token_list)
        m = module.model
        x = sample.to(module.device).unsqueeze(0)
        # same order as ModelModule.forward(): frontend -> proj_encoder -> encoder
        if hasattr(m, "frontend"):
            x = m.frontend(x)
        if hasattr(m, "proj_encoder"):
            x = m.proj_encoder(x)
        enc, _ = m.encoder(x, None)
        enc = enc.squeeze(0)
        hyps = module._nbest_bs(enc)[:k]

        out = []
        for h in hyps:
            d = h.asdict()
            ids = torch.tensor([int(t) for t in d["yseq"][1:]])
            text = module.text_transform.post_process(ids).replace("<eos>", "").strip()
            sc = d.get("scores", {})
            out.append({
                "text": text,
                "score": float(d["score"]),
                "n_tok": max(len(d["yseq"]) - 1, 1),
                "dec": float(sc.get("decoder", 0.0)),
                "ctc": float(sc.get("ctc", 0.0)),
            })
    return out


def nbest_features(nbest):
    """
    Method A confidence scores (higher = more confident)
      confA_logp   : length-normalized log-prob of the best hypothesis
      confA_margin : (best - second best) score per token
      confA_nbest  : 1 - mean difference between best and other N-best
    """
    if not nbest:
        return {"confA_logp": -99.0, "confA_margin": 0.0, "confA_nbest": 0.0}
    top = nbest[0]
    logp = top["score"] / top["n_tok"]
    if len(nbest) > 1:
        margin = (top["score"] - nbest[1]["score"]) / top["n_tok"]
        agree = 1.0 - sum(text_disagreement(top["text"], h["text"])
                          for h in nbest[1:]) / (len(nbest) - 1)
    else:
        margin, agree = 0.0, 1.0
    if not normalize(top["text"]):          # empty output -> not confident
        logp, agree = -99.0, 0.0
    return {"confA_logp": logp, "confA_margin": margin, "confA_nbest": agree}
