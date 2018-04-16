import mxnet as mx
from mxnet import nd

# a b c
# d e f
# g h i
# a(ei − fh) − b(di − fg) + c(dh − eg)
def compute_determinant(A):
    return A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])            - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])            + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])

# A shape is (N, P, 3, 3)
# return shape is (N, P, 3)
def compute_eigenvals(A):
    A_11 = A[:, :, 0, 0]  # (N, P)
    A_12 = A[:, :, 0, 1]
    A_13 = A[:, :, 0, 2]
    A_22 = A[:, :, 1, 1]
    A_23 = A[:, :, 1, 2]
    A_33 = A[:, :, 2, 2]
    I = nd.eye(3)
    p1 = nd.square(A_12) + nd.square(A_13) + nd.square(A_23)  # (N, P)
    q = (A_11 + A_22 + A_33) / 3  # (N, P)
    p2 = nd.square(A_11 - q) + nd.square(A_22 - q) + nd.square(A_33 - q) + 2 * p1  # (N, P)
    p = nd.sqrt(p2 / 6) + 1e-8  # (N, P)
    N = A.shape[0]
    q_4d = nd.reshape(q, (N, -1, 1, 1))  # (N, P, 1, 1)
    p_4d = nd.reshape(p, (N, -1, 1, 1))
    B = (1 / p_4d) * (A - q_4d * I)  # (N, P, 3, 3)
    r = nd.clip(compute_determinant(B) / 2, -1, 1)  # (N, P)
    phi = nd.arccos(r) / 3  # (N, P)
    eig1 = q + 2 * p * nd.cos(phi)  # (N, P)
    eig3 = q + 2 * p * nd.cos(phi + (2 * math.pi / 3))
    eig2 = 3 * q - eig1 - eig3
    return nd.abs(nd.stack([eig1, eig2, eig3], axis=2)) # (N, P, 3)

# P shape is (N, P, 3), N shape is (N, P, K, 3)
# return shape is (N, P)
def compute_curvature(nn_pts):
    nn_pts_mean = nd.mean(nn_pts, axis=2, keepdims=True)  # (N, P, 1, 3)
    nn_pts_demean = nn_pts - nn_pts_mean  # (N, P, K, 3)
    nn_pts_NPK31 = nd.expand_dims(nn_pts_demean, axis=-1)
    covariance_matrix = nd.batch_dot(nn_pts_NPK31, nn_pts_NPK31, transpose_b=True)  # (N, P, K, 3, 3)
    covariance_matrix_mean = nd.mean(covariance_matrix, axis=2, keepdims=False)  # (N, P, 3, 3)
    eigvals = compute_eigenvals(covariance_matrix_mean)  # (N, P, 3)
    curvature = nd.min(eigvals, axis=-1) / (nd.sum(eigvals, axis=-1) + 1e-8)
    return curvature

def curvature_based_sample(nn_pts, k):
    curvature = compute_curvature(nn_pts)
    point_indices = nd.topk(curvature, axis=-1, k=k, ret_typ='indices')

    pts_shape = nn_pts.shape
    batch_size = pts_shape[0]
    batch_indices = nd.tile(nd.reshape(nd.arange(batch_size), (-1, 1, 1)), (1, k, 1))
    indices = nd.concat(batch_indices, nd.expand_dims(point_indices, axis=2), dim=2)
    return indices