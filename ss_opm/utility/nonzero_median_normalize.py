import numpy as np
import scipy.sparse


# def median_normalize(values, ignore_zero=True, log=False):
#     tmp_values = values.copy()
#     if isinstance(values, scipy.sparse.csr_matrix):
#         tmp_values = tmp_values.toarray()
#     if ignore_zero:
#         tmp_values[tmp_values == 0] = np.nan
#     nonzaero_median = np.nanquantile(tmp_values, q=0.5, axis=1).astype(values.dtype)
#     # print("nonzaero_median", nonzaero_median)
#     if log:
#         ret = values - nonzaero_median[:, None]
#     else:
#         ret = values / nonzaero_median[:, None]
#     return ret


def median_normalize(values, ignore_zero=True, log=False):
    """
    Median-normalize rows of `values` in-place to avoid huge temporary arrays.
    If you need to keep `values` unchanged, call this as:
        normalized = median_normalize(values.copy(), ...)
    """
    # If sparse, convert once to dense (you can’t avoid the 10GB here if you want dense output)
    if isinstance(values, scipy.sparse.csr_matrix):
        values = values.toarray()

    # Ensure float32 (or float16 if you really need to save more memory)
    values = values.astype(np.float32, copy=False)

    if ignore_zero:
        # Mark zeros as NaN in-place
        zero_mask = (values == 0)
        values[zero_mask] = np.nan

        # Row-wise median ignoring NaNs
        nonzero_median = np.nanquantile(values, q=0.5, axis=1).astype(values.dtype)

        # Put zeros back
        values[zero_mask] = 0.0
    else:
        nonzero_median = np.quantile(values, q=0.5, axis=1).astype(values.dtype)

    # Avoid creating a new big array: normalize in-place
    # shape of nonzero_median[:, None] is (N, 1), but broadcasting is handled per-row inside the ufunc
    if log:
        values -= nonzero_median[:, None]
    else:
        values /= nonzero_median[:, None]

    return values


def row_quantile_normalize(values, q=0.5):
    ret_data = np.zeros_like(values.data)
    for i in range(values.shape[0]):
        row_values = values.data[values.indptr[i] : values.indptr[i + 1]]
        q_value = np.quantile(row_values, q=q)
        row_values /= q_value
        ret_data[values.indptr[i] : values.indptr[i + 1]] = row_values
    return scipy.sparse.csr_matrix((ret_data, values.indices, values.indptr), values.shape)
