import numpy as np
import scipy
import scipy.optimize as opt
import scipy.signal as sg


def build_score_matrix(x, delta_max):
    H, W = x.shape
    rows = np.arange(H)
    deltas = np.arange(-delta_max, delta_max + 1)
    y = np.zeros((len(deltas), W))

    for i, delta in enumerate(deltas):
        offset = np.round(rows * (delta / H)).astype(int)
        change_idx = np.flatnonzero(np.diff(offset)) + 1
        bounds = np.concatenate(([0], change_idx, [H]))

        sums = np.zeros(W)
        counts = np.zeros(W)
        for b in range(len(bounds) - 1):
            current_bound = bounds[b]
            next_bound = bounds[b + 1]
            k = int(offset[current_bound])
            row_sum = x[current_bound:next_bound].sum(axis=0)
            n = next_bound - current_bound
            if k >= 0:
                sums[0 : W - k] += row_sum[k:W]
                counts[0 : W - k] += n
            else:
                sums[-k:W] += row_sum[0 : W + k]
                counts[-k:W] += n

        y[i] = np.divide(sums, counts, out=np.zeros(W), where=counts > 0)
    return y


def my_find_min(y, delta_max, min_lenticule_width, max_lenticule_width, w):
    _, W = y.shape
    minimum_list = []
    valleys_depth_list = []

    y_full = y.copy()

    for delta in range(-delta_max, delta_max + 1):
        z = y_full[delta + delta_max]

        valleys = []
        valley_one = W // 2 + np.argmin(z[W // 2 : W // 2 + max_lenticule_width])

        valleys.append(valley_one)
        current_valley = valley_one
        while current_valley + min_lenticule_width < W:
            lb = int(current_valley) + min_lenticule_width
            ub = min(int(current_valley) + max_lenticule_width, W)
            j = np.argmin(z[lb:ub])
            current_valley = j + lb
            valleys.append(current_valley)

        current_valley = valley_one
        while current_valley - min_lenticule_width >= 0:
            lb = max(int(current_valley) - max_lenticule_width, 0)
            ub = int(current_valley) - min_lenticule_width
            if lb != ub:
                j = np.argmin(z[lb:ub])
            else:
                j = 0
            current_valley = j + lb
            valleys.append(current_valley)

        valleys = np.sort(valleys)

        valleys_depth = np.sum(z[np.round(valleys).astype(int)])

        minimum_list.append(np.array(valleys))
        valleys_depth_list.append(valleys_depth)

    idx = np.argmin(np.array(valleys_depth_list))
    lenticules_location_bottom = minimum_list[idx]
    lenticules_location_top = minimum_list[idx] + (idx - delta_max)

    return (
        lenticules_location_bottom,
        lenticules_location_top,
        (idx - delta_max),
    )


def optimize_locations(
    y,
    delta_max,
    min_lenticule_width,
    max_lenticule_width,
    w,
    lambda1,
    lambda2,
):
    initial_bottom, initial_top, _ = my_find_min(
        y, delta_max, min_lenticule_width, max_lenticule_width, w
    )

    M = len(initial_bottom) - 1

    x0 = np.concatenate([initial_bottom[1:], initial_top[1:]], axis=0)
    f = scipy.interpolate.RectBivariateSpline(
        np.arange(0, y.shape[0]) - delta_max, np.arange(0, y.shape[1]), y
    )

    D = -np.eye(M - 1, M=M, k=0) + np.eye(M - 1, M=M, k=1)
    e = np.ones((1, M))
    H = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), M - 2, M)

    def obj(x, f):
        bottom = x[0:M]
        top = x[M : 2 * M]
        delta = top - bottom

        val = f(delta, bottom, grid=False)
        val_tot = np.sum(val)

        penalty1_bottom = (D @ bottom - w).T @ (D @ bottom - w)
        penalty1_top = (D @ top - w).T @ (D @ top - w)

        penalty2_bottom = (H @ bottom).T @ (H @ bottom)
        penalty2_top = (H @ top).T @ (H @ top)

        return (
            val_tot
            + lambda2 * (penalty2_bottom + penalty2_top)
            + lambda1 * (penalty1_top + penalty1_bottom)
        )

    def jac(x, f):
        bottom = x[0:M]
        top = x[M : 2 * M]
        delta = top - bottom

        df_delta = f(delta, bottom, grid=False, dx=1, dy=0)
        df_bottom = f(delta, bottom, grid=False, dx=0, dy=1)

        df_top = df_delta
        df_bottom = df_bottom - df_delta

        gradbottom = lambda2 * (H @ bottom).T @ (H)
        gradtop = lambda2 * (H @ top).T @ (H)

        gradbottom += lambda1 * ((D @ bottom - w).T @ (D))
        gradtop += lambda1 * (D @ top - w).T @ (D)

        grad = np.concatenate([df_bottom + gradbottom, df_top + gradtop], axis=0)
        return grad

    res = opt.minimize(obj, x0, args=(f), method="L-BFGS-B", jac=jac)

    x = res.x

    bottom = initial_bottom.astype(float)
    bottom[1:] = x[0:M]
    top = initial_top.astype(float)
    top[1:] = x[M : 2 * M]
    return bottom, top


def reconstruct_boundaries(
    lenticules_location_bottom,
    lenticules_location_top,
    size,
):
    H, W = size
    z = np.zeros((H, W))

    for bottom, top in zip(lenticules_location_bottom, lenticules_location_top):

        idx_row = np.arange(0, H)
        idx_col = bottom + idx_row * (top - bottom) / H
        # idx_col_floor = np.floor(idx_col).astype(int)
        # idx_col_ceil = np.ceil(idx_col).astype(int)
        idx_col_floor = np.round(idx_col).astype(int)
        idx_col_ceil = np.round(idx_col).astype(int)
        idx_row = idx_row.astype(int)

        mask = np.logical_and(idx_col_floor >= 0, idx_col_ceil < W)
        idx_col = idx_col[mask]
        idx_col_floor = idx_col_floor[mask]
        idx_col_ceil = idx_col_ceil[mask]
        idx_row = idx_row[mask]

        z[idx_row, idx_col_floor] = idx_col_ceil - idx_col
        z[idx_row, idx_col_ceil] = idx_col - idx_col_floor

        mask2 = idx_col_floor == idx_col_ceil
        z[idx_row[mask2], idx_col_ceil[mask2]] = 1.0

    return z
