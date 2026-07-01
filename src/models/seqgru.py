#!/usr/bin/env python3
"""Global GRU encoder-decoder, directly trained on the weekly product series with
covariates. A decorrelated learner vs the foundation models: learns this data's
per-product seasonality + covariate response end-to-end, where chronos/timesfm
transfer. Multi-origin windows; pinball loss at a high quantile (WAPE wants slight
over-forecast). Weekly grain, 13-week direct decode.

Train + predict in one call via `fit_predict(valid_date, series_df, ...)`.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..core_target import PRODUCT_COLS

EVENTS = ["prime_day", "big_deals", "black_fri", "cyber_mon", "christmas", "ny_day"]
H = 13
CTX = 104  # context weeks

# ---- feature builders -------------------------------------------------------
def _woy_feats(weeks: pd.DatetimeIndex):
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    ang = 2 * np.pi * woy / 52.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1)

def _prep_product(g, scale):
    """past-feature matrix (T, Fpast) and target vector (T,) normalized by scale.
    tn keeps NaN where the week had no in-stock demand (masked in the loss); the
    feature copy fills those with 0."""
    t = g["target"].to_numpy(dtype=np.float32)
    tn = np.log1p(np.clip(t, 0, None) / scale)  # NaN preserved for masking
    tn_feat = np.nan_to_num(tn, nan=0.0)
    price = np.log1p(np.nan_to_num(g["avg_price_paid"].to_numpy(dtype=np.float32), nan=0.0))
    bb = np.nan_to_num(g["buybox_pct"].to_numpy(dtype=np.float32), nan=0.0)
    ad = np.log1p(np.nan_to_num(g["ad_spend"].to_numpy(dtype=np.float32), nan=0.0))
    oos = np.nan_to_num(g["oos"].to_numpy(dtype=np.float32), nan=0.0)
    promo = np.nan_to_num(g["promo_pct_off"].to_numpy(dtype=np.float32), nan=0.0)
    ev = g[EVENTS].to_numpy(dtype=np.float32)
    woy = _woy_feats(pd.DatetimeIndex(g["week_start"]))
    past = np.concatenate([tn_feat[:, None], price[:, None], bb[:, None], ad[:, None],
                           oos[:, None], promo[:, None], ev, woy], axis=1)
    return past, tn  # tn keeps NaN for masking

def _future_feats(weeks, promo, ev):
    woy = _woy_feats(pd.DatetimeIndex(weeks))
    promo = np.nan_to_num(promo, nan=0.0)
    return np.concatenate([promo[:, None], np.nan_to_num(ev), woy], axis=1).astype(np.float32)

NPAST = 1 + 4 + 1 + len(EVENTS) + 2  # tn, price/bb/ad/oos, promo, events, woy(2)
NFUT = 1 + len(EVENTS) + 2           # promo, events, woy(2)

# ---- model ------------------------------------------------------------------
class SeqGRU(nn.Module):
    def __init__(self, n_mkt, n_partner, hid=128, emb=16, layers=1, dropout=0.0):
        super().__init__()
        self.mkt = nn.Embedding(n_mkt, emb)
        self.par = nn.Embedding(n_partner, emb)
        drop = dropout if layers > 1 else 0.0
        self.enc = nn.GRU(NPAST, hid, num_layers=layers, batch_first=True, dropout=drop)
        self.dec = nn.GRU(NFUT + 2 * emb, hid, num_layers=layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, past, mkt_idx, par_idx, fut):
        _, h = self.enc(past)               # h: (layers,B,hid)
        e = torch.cat([self.mkt(mkt_idx), self.par(par_idx)], -1)  # (B,2emb)
        fe = torch.cat([fut, e[:, None, :].expand(-1, fut.shape[1], -1)], -1)
        out, _ = self.dec(fe, h)
        return self.head(out).squeeze(-1)   # (B,H) normalized log1p


def pinball(pred, target, q, mask):
    e = torch.nan_to_num(target) - pred
    loss = torch.maximum(q * e, (q - 1) * e) * mask
    return loss.sum() / mask.sum().clamp(min=1)


# ---- data assembly ----------------------------------------------------------
def _build(series, valid_date, train=True, max_origins=12):
    valid = pd.Timestamp(valid_date)
    series = series.sort_values(PRODUCT_COLS + ["week_start"])
    mkts = {m: i for i, m in enumerate(sorted(series["marketplace_id"].unique()))}
    pars = {p: i for i, p in enumerate(sorted(series["partner_id"].unique()))}
    samples = []  # dicts
    meta = []     # for inference: keys + scale
    for keyvals, g in series.groupby(PRODUCT_COLS, sort=False):
        gh = g[g["week_start"] < valid]
        if len(gh) < 8:
            if not train:
                meta.append((keyvals, max(1.0, gh["target"].mean() if len(gh) else 1.0), None, None))
            continue
        # per-product scale = mean of last 52 in-stock weeks (>0)
        recent = gh["target"].to_numpy(dtype=np.float32)[-52:]
        scale = max(1.0, float(np.mean(recent[recent > 0])) if (recent > 0).any() else 1.0)
        past_full, tn_full = _prep_product(gh, scale)
        weeks = pd.DatetimeIndex(gh["week_start"])
        if train:
            # origins: positions o with >=13 future weeks available, slide from the end
            last = len(gh) - H
            if last < 4:
                continue
            origins = list(range(last, 3, -max(1, (last - 3) // max_origins)))[:max_origins]
            for o in origins:
                ctx = past_full[max(0, o - CTX):o]
                fut_weeks = weeks[o:o + H]
                fut = _future_feats(fut_weeks, gh["promo_pct_off"].to_numpy(np.float32)[o:o + H],
                                    gh[EVENTS].to_numpy(np.float32)[o:o + H])
                tgt = tn_full[o:o + H]
                samples.append(dict(ctx=ctx, fut=fut, tgt=tgt,
                                    mkt=mkts[keyvals[0]], par=pars[keyvals[1]]))
        else:
            ctx = past_full[-CTX:]
            fut_weeks = pd.date_range(valid, periods=H, freq="7D")
            fdf = g[g["week_start"] >= valid].set_index("week_start")
            promo = np.array([fdf["promo_pct_off"].get(w, 0.0) for w in fut_weeks], np.float32)
            promo = np.nan_to_num(promo, nan=0.0)
            evf = np.zeros((H, len(EVENTS)), np.float32)
            for i, w in enumerate(fut_weeks):
                if w in fdf.index:
                    evf[i] = np.nan_to_num(fdf.loc[w, EVENTS].to_numpy(np.float32))
            fut = _future_feats(fut_weeks, promo, evf)
            samples.append(dict(ctx=ctx, fut=fut, mkt=mkts[keyvals[0]], par=pars[keyvals[1]]))
            meta.append((keyvals, scale, fut_weeks, len(samples) - 1))
    return samples, meta, len(mkts), len(pars)


def _pad_ctx(ctxs):
    B = len(ctxs)
    out = np.zeros((B, CTX, NPAST), np.float32)
    for i, c in enumerate(ctxs):
        L = min(len(c), CTX)
        if L:
            out[i, CTX - L:] = c[-L:]
    return out


def fit_predict(valid_date, series, quantile=0.55, epochs=12, lr=1e-3,
                batch=256, device=None, seed=0, hid=128, layers=1, dropout=0.0,
                weight_decay=0.0, origins=12, cosine=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, _, nm, npn = _build(series, valid_date, train=True, max_origins=origins)
    print(f"train samples: {len(tr):,}  mkts {nm} partners {npn}", flush=True)
    model = SeqGRU(nm, npn, hid=hid, layers=layers, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    ctx = _pad_ctx([s["ctx"] for s in tr])
    fut = np.stack([s["fut"] for s in tr]); tgt = np.stack([s["tgt"] for s in tr])
    mkt = np.array([s["mkt"] for s in tr]); par = np.array([s["par"] for s in tr])
    ctx_t = torch.tensor(ctx); fut_t = torch.tensor(fut); tgt_t = torch.tensor(tgt)
    mask_t = torch.tensor(~np.isnan(tgt), dtype=torch.float32)
    mkt_t = torch.tensor(mkt); par_t = torch.tensor(par)
    N = len(tr)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(N); tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            p = model(ctx_t[idx].to(dev), mkt_t[idx].to(dev), par_t[idx].to(dev), fut_t[idx].to(dev))
            loss = pinball(p, tgt_t[idx].to(dev), quantile, mask_t[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if sched is not None:
            sched.step()
        print(f"  epoch {ep+1}/{epochs} pinball {tot/N:.4f}", flush=True)
    # inference (batched over products; GRU sequences are independent so this is
    # numerically identical to a per-product loop, just much faster)
    model.eval()
    te, meta, _, _ = _build(series, valid_date, train=False)
    items = [(kv, scale, fw, sidx) for kv, scale, fw, sidx in meta if sidx is not None]
    ctx_t = torch.tensor(_pad_ctx([te[s[3]]["ctx"] for s in items]))
    fut_t = torch.tensor(np.stack([te[s[3]]["fut"] for s in items]))
    mkt_t = torch.tensor(np.array([te[s[3]]["mkt"] for s in items]))
    par_t = torch.tensor(np.array([te[s[3]]["par"] for s in items]))
    preds = np.empty((len(items), H), np.float32)
    with torch.no_grad():
        for i in range(0, len(items), 2048):
            sl = slice(i, i + 2048)
            preds[sl] = model(ctx_t[sl].to(dev), mkt_t[sl].to(dev),
                              par_t[sl].to(dev), fut_t[sl].to(dev)).cpu().numpy()
    rows = []
    for j, (keyvals, scale, fut_weeks, _) in enumerate(items):
        yhat = np.clip(np.expm1(preds[j]) * scale, 0, None)
        for h in range(H):
            rows.append((keyvals[0], keyvals[1], keyvals[2], h + 1, fut_weeks[h], float(yhat[h])))
    out = pd.DataFrame(rows, columns=PRODUCT_COLS + ["horizon_week", "horizon_date", "forecast"])
    return out


def ensemble_predict(valid_date, series, seeds=3, **kw):
    """Average `seeds` independently-seeded fits (variance reduction)."""
    preds = [fit_predict(valid_date, series, seed=s, **kw) for s in range(seeds)]
    if seeds == 1:
        return preds[0]
    keys = PRODUCT_COLS + ["horizon_week", "horizon_date"]
    return pd.concat(preds).groupby(keys, as_index=False)["forecast"].mean()
