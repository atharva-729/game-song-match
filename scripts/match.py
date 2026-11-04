"""Reference matcher. site/match.js must give identical results, so every step here is written the
way the JS does it: sequential sums (no numpy pairwise summation), the same interpolation, the same
DTW tie-breaking (diagonal, then up, then left; switch only when strictly smaller).

normalise: resample linearly to N=200 points, z-score (population std); std < 1e-6 -> unmatchable.
distance:  DTW with a Sakoe-Chiba band of +-W=10, squared difference, divided by path length.
score:     distance * (1 - 0.4 * famous). The README's flat "- 0.05" let famous games beat a game's own
           curve once 8,489 games packed neighbours within 0.03; a proportional 40% nudge never can.
"""
import math
import numpy as np

N, W, FAMOUS_BONUS, QSCALE = 200, 10, 0.4, 32.0


def resample(x, n=N):
    x = [float(v) for v in x]
    m = len(x)
    if m == 1:
        return [x[0]] * n
    out = []
    for k in range(n):
        t = k * (m - 1) / (n - 1)
        i = int(math.floor(t))
        if i >= m - 1:
            out.append(x[m - 1])
        else:
            f = t - i
            out.append(x[i] + (x[i + 1] - x[i]) * f)
    return out


def zscore(x):
    n = len(x)
    s = 0.0
    for v in x:
        s += v
    mean = s / n
    ss = 0.0
    for v in x:
        ss += (v - mean) * (v - mean)
    std = math.sqrt(ss / n)
    if std < 1e-6:
        return None
    return [(v - mean) / std for v in x]


def normalise(x):
    return zscore(resample(x))


def quantise(z):
    return np.clip(np.round(np.asarray(z) * QSCALE), -127, 127).astype(np.int8)


def dequantise(q):
    return np.asarray(q, dtype=np.float64) / QSCALE


def dtw_batch(a, B):
    """a: (N,) song curve. B: (M, N) game curves. Returns (M,) banded DTW distance / path length."""
    a = np.asarray(a, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    M, K = B.shape[0], 2 * W + 1
    INF = np.inf
    prevD = np.full((M, K), INF); prevL = np.zeros((M, K))
    for i in range(N):
        curD = np.full((M, K), INF); curL = np.zeros((M, K))
        for k in range(K):
            j = i + k - W
            if j < 0 or j >= N:
                continue
            d = a[i] - B[:, j]
            c = d * d
            if i == 0 and j == 0:
                curD[:, k] = c; curL[:, k] = 1
                continue
            bestD = np.full(M, INF); bestL = np.zeros(M)
            if i > 0 and j > 0:                      # diagonal (i-1, j-1): same band slot in previous row
                bestD = prevD[:, k].copy(); bestL = prevL[:, k].copy()
            if i > 0 and k + 1 < K:                  # up (i-1, j)
                m = prevD[:, k + 1] < bestD
                bestD = np.where(m, prevD[:, k + 1], bestD); bestL = np.where(m, prevL[:, k + 1], bestL)
            if j > 0 and k - 1 >= 0:                 # left (i, j-1)
                m = curD[:, k - 1] < bestD
                bestD = np.where(m, curD[:, k - 1], bestD); bestL = np.where(m, curL[:, k - 1], bestL)
            curD[:, k] = c + bestD; curL[:, k] = bestL + 1
        prevD, prevL = curD, curL
    return prevD[:, W] / prevL[:, W]


def dtw_path(a, b):
    """Single pair, same recurrence as dtw_batch, plus the warping path [(i, j), ...] from (0,0)."""
    K = 2 * W + 1
    D = [[math.inf] * K for _ in range(N)]
    L = [[0] * K for _ in range(N)]
    P = [[-1] * K for _ in range(N)]  # 0 diag, 1 up, 2 left
    for i in range(N):
        for k in range(K):
            j = i + k - W
            if j < 0 or j >= N:
                continue
            c = (a[i] - b[j]) * (a[i] - b[j])
            if i == 0 and j == 0:
                D[i][k], L[i][k] = c, 1
                continue
            bd, bl, bp = math.inf, 0, -1
            if i > 0 and j > 0:
                bd, bl, bp = D[i - 1][k], L[i - 1][k], 0
            if i > 0 and k + 1 < K and D[i - 1][k + 1] < bd:
                bd, bl, bp = D[i - 1][k + 1], L[i - 1][k + 1], 1
            if j > 0 and k - 1 >= 0 and D[i][k - 1] < bd:
                bd, bl, bp = D[i][k - 1], L[i][k - 1], 2
            D[i][k], L[i][k], P[i][k] = c + bd, bl + 1, bp
    path, i, k = [], N - 1, W
    while True:
        j = i + k - W
        path.append((i, j))
        p = P[i][k]
        if p == -1:
            break
        if p == 0:
            i -= 1
        elif p == 1:
            i, k = i - 1, k + 1
        else:
            k -= 1
    return D[N - 1][W] / L[N - 1][W], path[::-1]


def search(song, Q, meta, scope="both", top=5):
    """song: raw energy curve (any length). Q: (M, N) int8 index. meta: list of dicts with league, famous.
    Returns dict with top (list of (idx, score, dist)), best_nfl, best_ipl."""
    a = normalise(song)
    if a is None:
        return None
    dist = dtw_batch(a, dequantise(Q))
    fam = np.array([1.0 if m["famous"] else 0.0 for m in meta])
    score = dist * (1 - FAMOUS_BONUS * fam)
    league = np.array([m["league"] for m in meta])
    allowed = np.ones(len(meta), bool) if scope == "both" else (league == scope.upper())
    order = [int(i) for i in np.lexsort((np.arange(len(meta)), score)) if allowed[i]]
    res = {"top": [(i, float(score[i]), float(dist[i])) for i in order[:top]]}
    for lg in ("NFL", "IPL"):
        best = next((i for i in order if league[i] == lg), None)
        res[f"best_{lg.lower()}"] = None if best is None else (best, float(score[best]), float(dist[best]))
    return res
