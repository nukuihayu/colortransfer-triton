"""Fused image operations and bounded-memory color transport kernels."""

import triton
import triton.language as tl


@triton.jit
def _colorfm_velocity(r, g, b, t, A0, A1, A2, A3, O0, O1, O2):
    z = (
        r[:, None] * A0[None, :]
        + g[:, None] * A1[None, :]
        + b[:, None] * A2[None, :]
        + t * A3[None, :]
    )
    activation = z * tl.sigmoid(z)
    return (
        tl.sum(activation * O0[None, :], 1),
        tl.sum(activation * O1[None, :], 1),
        tl.sum(activation * O2[None, :], 1),
    )


@triton.jit
def colorfm_integrate(
    X,
    W1,
    W2,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    STEPS: tl.constexpr,
    TIME: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    j = tl.arange(0, K)
    r = tl.load(X + i * 3, i < N, 0)
    g = tl.load(X + i * 3 + 1, i < N, 0)
    b = tl.load(X + i * 3 + 2, i < N, 0)
    a0 = tl.load(W1 + j * 4, j < H, 0)
    a1 = tl.load(W1 + j * 4 + 1, j < H, 0)
    a2 = tl.load(W1 + j * 4 + 2, j < H, 0)
    a3 = tl.load(W1 + j * 4 + 3, j < H, 0)
    o0 = tl.load(W2 + j, j < H, 0)
    o1 = tl.load(W2 + H + j, j < H, 0)
    o2 = tl.load(W2 + 2 * H + j, j < H, 0)
    dt: tl.constexpr = TIME / STEPS
    for step in range(STEPS):
        t = step * dt
        v0, v1, v2 = _colorfm_velocity(r, g, b, t, a0, a1, a2, a3, o0, o1, o2)
        u0, u1, u2 = _colorfm_velocity(
            r + dt * 0.5 * v0,
            g + dt * 0.5 * v1,
            b + dt * 0.5 * v2,
            t + dt * 0.5,
            a0,
            a1,
            a2,
            a3,
            o0,
            o1,
            o2,
        )
        r += dt * u0
        g += dt * u1
        b += dt * u2
    tl.store(Y + i * 3, r, i < N)
    tl.store(Y + i * 3 + 1, g, i < N)
    tl.store(Y + i * 3 + 2, b, i < N)


@triton.jit
def moments_part(X, P, PIXELS: tl.constexpr, CHUNKS: tl.constexpr, B: tl.constexpr):
    plane = tl.program_id(0)
    chunk = tl.program_id(1)
    i = chunk * B + tl.arange(0, B)
    count = tl.minimum(B, PIXELS - chunk * B)
    x = tl.load(X + plane * PIXELS + i, i < PIXELS, 0).to(tl.float32)
    mean = tl.sum(x, 0) / count
    m2 = tl.sum(tl.where(i < PIXELS, (x - mean) * (x - mean), 0.0), 0)
    tl.store(P + plane * CHUNKS * 2 + chunk * 2, mean)
    tl.store(P + plane * CHUNKS * 2 + chunk * 2 + 1, m2)


@triton.jit
def moments_finish(
    P,
    S,
    PIXELS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CORRECTION: tl.constexpr,
    CHUNK: tl.constexpr,
    B: tl.constexpr,
):
    plane = tl.program_id(0)
    i = tl.arange(0, B)
    count = tl.minimum(CHUNK, tl.maximum(PIXELS - i * CHUNK, 0))
    mean = tl.load(P + plane * CHUNKS * 2 + i * 2, i < CHUNKS, 0)
    m2 = tl.load(P + plane * CHUNKS * 2 + i * 2 + 1, i < CHUNKS, 0)
    avg = tl.sum(mean * count, 0) / PIXELS
    var = tl.sum(m2 + (mean - avg) * (mean - avg) * count, 0) / max(PIXELS - CORRECTION, 1)
    tl.store(S + plane * 2, avg)
    tl.store(S + plane * 2 + 1, var)


@triton.jit
def adain_apply(
    X,
    Y,
    S,
    R,
    PIXELS: tl.constexpr,
    TOTAL: tl.constexpr,
    REF_BATCH: tl.constexpr,
    EPS: tl.constexpr,
    STRENGTH: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    at = tl.program_id(0) * B + tl.arange(0, B)
    plane = tl.program_id(1)
    i = plane * PIXELS + at
    rp = plane % 3 if REF_BATCH == 1 else plane
    mean = tl.load(S + plane * 2)
    var = tl.load(S + plane * 2 + 1)
    rm = tl.load(R + rp * 2)
    rv = tl.load(R + rp * 2 + 1)
    x = tl.load(X + i, at < PIXELS, 0).to(tl.float32)
    y = (x - mean) * tl.sqrt((rv + EPS) / (var + EPS)) + rm
    y = x + STRENGTH * (y - x)
    if CLAMP:
        y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    tl.store(Y + i, y, at < PIXELS)


@triton.jit
def sample_image(
    X, Random, Y, P: tl.constexpr, M: tl.constexpr, BATCH: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if M == P:
        at = i
    else:
        lo = i.to(tl.int64) * P // M
        hi = (i.to(tl.int64) + 1) * P // M
        jitter = tl.load(Random + i, i < M, 0)
        at = (lo + (jitter * (hi - lo).to(tl.float32)).to(tl.int64)).to(tl.int32)
    for c in tl.static_range(3):
        value = tl.load(X + (BATCH * 3 + c) * P + at, i < M, 0).to(tl.float32)
        tl.store(Y + i * 3 + c, value, i < M)


@triton.jit(do_not_specialize=["STEP"])
def project(X, D, Y, M: tl.constexpr, STEP, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r = tl.load(X + i * 3, i < M, 0)
    g = tl.load(X + i * 3 + 1, i < M, 0)
    b = tl.load(X + i * 3 + 2, i < M, 0)
    dr = tl.load(D + STEP * 3)
    dg = tl.load(D + STEP * 3 + 1)
    db = tl.load(D + STEP * 3 + 2)
    tl.store(Y + i, r * dr + g * dg + b * db, i < M)


@triton.jit(do_not_specialize=["STEP"])
def project_sort(X, D, Y, M: tl.constexpr, STEP, B: tl.constexpr, ALL: tl.constexpr = False):
    # All reference directions are independent; source directions evolve in order.
    step = tl.program_id(0) if ALL else STEP
    offset = step * M if ALL else 0
    i = tl.arange(0, B)
    r = tl.load(X + i * 3, i < M, 0)
    g = tl.load(X + i * 3 + 1, i < M, 0)
    b = tl.load(X + i * 3 + 2, i < M, 0)
    dr = tl.load(D + step * 3)
    dg = tl.load(D + step * 3 + 1)
    db = tl.load(D + step * 3 + 2)
    p = tl.where(i < M, r * dr + g * dg + b * db, float("inf"))
    ordered = tl.sort(p, descending=False)
    tl.store(Y + offset + i, ordered, i < M)


@triton.jit(do_not_specialize=["STEP"])
def sliced_update(
    X,
    Y,
    S,
    R,
    D,
    RM,
    Q: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    STEP,
    B: tl.constexpr,
    PRECOMPUTED: tl.constexpr = False,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r = tl.load(X + i * 3, i < Q, 0)
    g = tl.load(X + i * 3 + 1, i < Q, 0)
    b = tl.load(X + i * 3 + 2, i < Q, 0)
    dr = tl.load(D + STEP * 3)
    dg = tl.load(D + STEP * 3 + 1)
    db = tl.load(D + STEP * 3 + 2)
    p = r * dr + g * dg + b * db
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), M, tl.int32)
    ulo = lo
    uhi = hi
    for _ in range((M + 1).bit_length()):
        mid = (lo + hi) // 2
        v = tl.load(S + mid, mid < M, float("inf"))
        go = (mid < M) & (v < p)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
        umid = (ulo + uhi) // 2
        uv = tl.load(S + umid, umid < M, float("inf"))
        ugo = (umid < M) & (uv <= p)
        ulo = tl.where(ugo, umid + 1, ulo)
        uhi = tl.where(ugo, uhi, umid)
    left = tl.minimum(tl.maximum(lo - 1, 0), M - 1)
    right = tl.minimum(lo, M - 1)
    a = tl.load(S + left)
    z = tl.load(S + right)
    fraction = tl.minimum(tl.maximum((p - a) / tl.maximum(z - a, 1e-20), 0.0), 1.0)
    rank = tl.where(ulo > lo, (lo + ulo - 1) * 0.5, left + fraction * (right - left))
    pos = tl.minimum(tl.maximum((rank + 0.5) * (N / M) - 0.5, 0.0), N - 1.0)
    j = pos.to(tl.int32)
    f = pos - j
    roffset = STEP * N if PRECOMPUTED else 0
    q0 = tl.load(R + roffset + j)
    q1 = tl.load(R + roffset + tl.minimum(j + 1, N - 1))
    target = q0 + f * (q1 - q0)
    first = tl.load(S)
    last = tl.load(S + M - 1)
    avg = tl.load(RM) * dr + tl.load(RM + 1) * dg + tl.load(RM + 2) * db
    delta = tl.where(first == last, avg - first, target - p)
    tl.store(Y + i * 3, r + delta * dr, i < Q)
    tl.store(Y + i * 3 + 1, g + delta * dg, i < Q)
    tl.store(Y + i * 3 + 2, b + delta * db, i < Q)


@triton.jit
def wavelet_pass(
    X,
    Ref,
    Low,
    Y,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    D: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
    RB: tl.constexpr,
    STRENGTH: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    plane = i // (H * W)
    y = (i // W) % H
    x = i % W
    acc = tl.full((B,), 0, tl.float32)
    for dy in tl.static_range(-1, 2):
        yy = tl.minimum(tl.maximum(y + dy * D, 0), H - 1)
        for dx in tl.static_range(-1, 2):
            xx = tl.minimum(tl.maximum(x + dx * D, 0), W - 1)
            address = (plane * H + yy) * W + xx
            if FIRST:
                rplane = plane % 3 if RB == 1 else plane
                a = tl.load(X + address, i < TOTAL, 0).to(tl.float32)
                r = tl.load(Ref + (rplane * H + yy) * W + xx, i < TOTAL, 0).to(tl.float32)
                value = r - a
            else:
                value = tl.load(Low + address, i < TOTAL, 0)
            weight = (0.5 if dy == 0 else 0.25) * (0.5 if dx == 0 else 0.25)
            acc = tl.fma(value, weight, acc)
    if LAST:
        original = tl.load(X + i, i < TOTAL, 0).to(tl.float32)
        acc = original + STRENGTH * acc
        if CLAMP:
            acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    tl.store(Y + i, acc, i < TOTAL)


@triton.jit
def blend_reference(
    X,
    R,
    Y,
    PIXELS: tl.constexpr,
    TOTAL: tl.constexpr,
    RB: tl.constexpr,
    STRENGTH: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ri = i % (3 * PIXELS) if RB == 1 else i
    x = tl.load(X + i, i < TOTAL, 0).to(tl.float32)
    r = tl.load(R + ri, i < TOTAL, 0).to(tl.float32)
    out = x + STRENGTH * (r - x)
    if CLAMP:
        out = tl.minimum(tl.maximum(out, 0.0), 1.0)
    tl.store(Y + i, out, i < TOTAL)


@triton.jit
def cost_matrix(
    X,
    Y,
    C,
    CT,
    M: tl.constexpr,
    N: tl.constexpr,
    PARTIAL: tl.constexpr,
    CS: tl.constexpr,
    CTS: tl.constexpr,
    B: tl.constexpr,
):
    rows = M + 1 if PARTIAL else M
    cols = N + 1 if PARTIAL else N
    i = tl.program_id(0) * B + tl.arange(0, B)
    r = i // cols
    c = i % cols
    total = rows * cols
    dist = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(3):
        x = tl.load(X + r * 3 + k, (i < total) & (r < M), 0)
        y = tl.load(Y + c * 3 + k, (i < total) & (c < N), 0)
        dist += (x - y) * (x - y)
    if PARTIAL:
        dist = tl.where((r == M) | (c == N), 0.0, dist)
        dist = tl.where((r == M) & (c == N), float("inf"), dist)
    tl.store(C + r * CS + c, dist, i < total)
    tl.store(CT + c * CTS + r, dist, i < total)


@triton.jit
def sinkhorn_step(
    C,
    V,
    U,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CS: tl.constexpr,
    REAL_ROWS: tl.constexpr,
    MASS: tl.constexpr,
    EPS: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, K)
    cost = tl.load(
        C + row[:, None] * CS + col[None, :],
        (row[:, None] < ROWS) & (col[None, :] < COLS),
        float("inf"),
    )
    v = tl.load(V + col, col < COLS, 0)
    logits = (v[None, :] - cost) / EPS
    peak = tl.max(logits, 1)
    # Invalid rows are padded with -inf and never stored.
    peak = tl.where(row < ROWS, peak, 0.0)
    lse = peak + tl.log(tl.sum(tl.exp(logits - peak[:, None]), 1))
    marginal = tl.where(row < REAL_ROWS, 1.0 / REAL_ROWS, 1.0 - MASS)
    value = EPS * (tl.log(marginal) - lse)
    tl.store(U + row, value, row < ROWS)


@triton.jit
def coupling(
    C,
    U,
    V,
    P,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CS: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // COLS
    col = i % COLS
    c = tl.load(C + row * CS + col, i < ROWS * COLS, float("inf"))
    u = tl.load(U + row, row < ROWS, 0)
    v = tl.load(V + col, col < COLS, 0)
    tl.store(P + i, tl.exp((u + v - c) / EPS), i < ROWS * COLS)


@triton.jit
def barycentric(
    Q,
    Y,
    V,
    Output,
    COUNT: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    PARTIAL: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q0 = tl.load(Q + i * 3, i < COUNT, 0)
    q1 = tl.load(Q + i * 3 + 1, i < COUNT, 0)
    q2 = tl.load(Q + i * 3 + 2, i < COUNT, 0)
    peak = tl.full((B,), float("-inf"), tl.float32)
    denom = tl.full((B,), 0, tl.float32)
    out0 = denom
    out1 = denom
    out2 = denom
    for offset in range(triton.cdiv(N, K)):
        j = offset * K + tl.arange(0, K)
        y0 = tl.load(Y + j * 3, j < N, 0)
        y1 = tl.load(Y + j * 3 + 1, j < N, 0)
        y2 = tl.load(Y + j * 3 + 2, j < N, 0)
        v = tl.load(V + j, j < N, 0)
        d0 = q0[:, None] - y0[None, :]
        d1 = q1[:, None] - y1[None, :]
        d2 = q2[:, None] - y2[None, :]
        c = d0 * d0 + d1 * d1 + d2 * d2
        logit = tl.where(j[None, :] < N, (v[None, :] - c) / EPS, float("-inf"))
        newpeak = tl.maximum(peak, tl.max(logit, 1))
        factor = tl.exp(peak - newpeak)
        w = tl.exp(logit - newpeak[:, None])
        denom = denom * factor + tl.sum(w, 1)
        out0 = out0 * factor + tl.sum(w * y0[None, :], 1)
        out1 = out1 * factor + tl.sum(w * y1[None, :], 1)
        out2 = out2 * factor + tl.sum(w * y2[None, :], 1)
        peak = newpeak
    if PARTIAL:
        dummy = tl.load(V + N) / EPS
        newpeak = tl.maximum(peak, dummy)
        factor = tl.exp(peak - newpeak)
        w = tl.exp(dummy - newpeak)
        denom = denom * factor + w
        out0 = out0 * factor + w * q0
        out1 = out1 * factor + w * q1
        out2 = out2 * factor + w * q2
    tl.store(Output + i * 3, out0 / denom, i < COUNT)
    tl.store(Output + i * 3 + 1, out1 / denom, i < COUNT)
    tl.store(Output + i * 3 + 2, out2 / denom, i < COUNT)


@triton.jit
def lut_apply(
    X,
    L,
    Y,
    P: tl.constexpr,
    TOTAL: tl.constexpr,
    S: tl.constexpr,
    LB: tl.constexpr,
    STRENGTH: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    batch = i // P
    pix = i % P
    r = tl.load(X + (batch * 3) * P + pix, i < TOTAL, 0).to(tl.float32)
    g = tl.load(X + (batch * 3 + 1) * P + pix, i < TOTAL, 0).to(tl.float32)
    b = tl.load(X + (batch * 3 + 2) * P + pix, i < TOTAL, 0).to(tl.float32)
    cr = tl.minimum(tl.maximum(r, 0.0), 1.0) * (S - 1)
    cg = tl.minimum(tl.maximum(g, 0.0), 1.0) * (S - 1)
    cb = tl.minimum(tl.maximum(b, 0.0), 1.0) * (S - 1)
    ir = tl.minimum(cr.to(tl.int32), S - 2)
    ig = tl.minimum(cg.to(tl.int32), S - 2)
    ib = tl.minimum(cb.to(tl.int32), S - 2)
    fr = cr - ir
    fg = cg - ig
    fb = cb - ib
    lbatch = 0 if LB == 1 else batch
    out0 = tl.full((B,), 0, tl.float32)
    out1 = out0
    out2 = out0
    for dr in tl.static_range(2):
        for dg in tl.static_range(2):
            for db in tl.static_range(2):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                address = ((lbatch * S + ir + dr) * S + ig + dg) * S + ib + db
                v0 = tl.load(L + address * 3, i < TOTAL, 0)
                v1 = tl.load(L + address * 3 + 1, i < TOTAL, 0)
                v2 = tl.load(L + address * 3 + 2, i < TOTAL, 0)
                out0 = tl.fma(w, v0, out0)
                out1 = tl.fma(w, v1, out1)
                out2 = tl.fma(w, v2, out2)
    out0 = r + STRENGTH * (out0 - r)
    out1 = g + STRENGTH * (out1 - g)
    out2 = b + STRENGTH * (out2 - b)
    if CLAMP:
        out0 = tl.minimum(tl.maximum(out0, 0.0), 1.0)
        out1 = tl.minimum(tl.maximum(out1, 0.0), 1.0)
        out2 = tl.minimum(tl.maximum(out2, 0.0), 1.0)
    tl.store(Y + batch * 3 * P + pix, out0, i < TOTAL)
    tl.store(Y + (batch * 3 + 1) * P + pix, out1, i < TOTAL)
    tl.store(Y + (batch * 3 + 2) * P + pix, out2, i < TOTAL)


@triton.jit
def _linear(x):
    value = (tl.maximum(x, 0.04045) + 0.055) / 1.055
    return tl.where(x <= 0.04045, x / 12.92, tl.exp(tl.log(value) * 2.4))


@triton.jit
def lut_pack(L, T, S: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    # The eight corners of each cell are contiguous for coalesced vector loads.
    i = tl.program_id(0) * B + tl.arange(0, B)
    cells: tl.constexpr = (S - 1) ** 3
    batch = i // (cells * 24)
    c = (i // 8) % 3
    corner = i % 8
    cell = (i // 24) % cells
    r = cell // ((S - 1) ** 2) + corner // 4
    g = (cell // (S - 1)) % (S - 1) + (corner // 2) % 2
    b = cell % (S - 1) + corner % 2
    v = tl.load(L + (((batch * S + r) * S + g) * S + b) * 3 + c, i < TOTAL, 0)
    tl.store(T + i, v, i < TOTAL)


@triton.jit
def lut_apply_packed(
    X,
    T,
    Y,
    P: tl.constexpr,
    S: tl.constexpr,
    LB: tl.constexpr,
    STRENGTH: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    batch = tl.program_id(1)
    r = tl.load(X + batch * 3 * P + i, i < P, 0).to(tl.float32)
    g = tl.load(X + (batch * 3 + 1) * P + i, i < P, 0).to(tl.float32)
    b = tl.load(X + (batch * 3 + 2) * P + i, i < P, 0).to(tl.float32)
    cr = tl.minimum(tl.maximum(r, 0.0), 1.0) * (S - 1)
    cg = tl.minimum(tl.maximum(g, 0.0), 1.0) * (S - 1)
    cb = tl.minimum(tl.maximum(b, 0.0), 1.0) * (S - 1)
    ir = tl.minimum(cr.to(tl.int32), S - 2)
    ig = tl.minimum(cg.to(tl.int32), S - 2)
    ib = tl.minimum(cb.to(tl.int32), S - 2)
    fr, fg, fb = cr - ir, cg - ig, cb - ib
    lbatch = 0 if LB == 1 else batch
    cell = ((lbatch * (S - 1) + ir) * (S - 1) + ig) * (S - 1) + ib
    corner = tl.arange(0, 8)
    weight = (
        tl.where(corner[None, :] // 4 != 0, fr[:, None], 1 - fr[:, None])
        * tl.where((corner[None, :] // 2) % 2 != 0, fg[:, None], 1 - fg[:, None])
        * tl.where(corner[None, :] % 2 != 0, fb[:, None], 1 - fb[:, None])
    )
    for c in tl.static_range(3):
        value = tl.load(T + cell[:, None] * 24 + c * 8 + corner[None, :], i[:, None] < P, 0)
        out = tl.sum(value * weight, 1)
        original = r if c == 0 else (g if c == 1 else b)
        out = original + STRENGTH * (out - original)
        if CLAMP:
            out = tl.minimum(tl.maximum(out, 0.0), 1.0)
        tl.store(Y + (batch * 3 + c) * P + i, out, i < P)


@triton.jit
def _lab_f(x):
    return tl.where(
        x > (6.0 / 29.0) ** 3,
        tl.exp(tl.log(tl.maximum(x, (6.0 / 29.0) ** 3)) / 3.0),
        x / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0,
    )


@triton.jit
def rgb_lab(X, Y, P: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    n = i // P
    p = i % P
    r = _linear(tl.load(X + n * 3 * P + p, i < TOTAL, 0).to(tl.float32))
    g = _linear(tl.load(X + (n * 3 + 1) * P + p, i < TOTAL, 0).to(tl.float32))
    b = _linear(tl.load(X + (n * 3 + 2) * P + p, i < TOTAL, 0).to(tl.float32))
    xx = _lab_f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047)
    yy = _lab_f(0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
    zz = _lab_f((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883)
    tl.store(Y + n * 3 * P + p, 116 * yy - 16, i < TOTAL)
    tl.store(Y + (n * 3 + 1) * P + p, 500 * (xx - yy), i < TOTAL)
    tl.store(Y + (n * 3 + 2) * P + p, 200 * (yy - zz), i < TOTAL)
