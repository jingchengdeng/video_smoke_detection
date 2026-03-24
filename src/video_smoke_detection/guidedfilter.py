from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
from numpy.linalg import inv


R, G, B = 0, 1, 2


def boxfilter(image: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    result = np.zeros((height, width))

    sum_y = np.cumsum(image, axis=0)
    result[: radius + 1] = sum_y[radius : 2 * radius + 1]
    result[radius + 1 : height - radius] = (
        sum_y[2 * radius + 1 :] - sum_y[: height - 2 * radius - 1]
    )
    result[-radius:] = np.tile(sum_y[-1], (radius, 1)) - sum_y[
        height - 2 * radius - 1 : height - radius - 1
    ]

    sum_x = np.cumsum(result, axis=1)
    result[:, : radius + 1] = sum_x[:, radius : 2 * radius + 1]
    result[:, radius + 1 : width - radius] = (
        sum_x[:, 2 * radius + 1 :] - sum_x[:, : width - 2 * radius - 1]
    )
    result[:, -radius:] = np.tile(sum_x[:, -1][:, None], (1, radius)) - sum_x[
        :, width - 2 * radius - 1 : width - radius - 1
    ]
    return result


def guided_filter(image: np.ndarray, guided: np.ndarray, radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    height, width = guided.shape
    base = boxfilter(np.ones((height, width)), radius)
    means = [boxfilter(image[:, :, i], radius) / base for i in range(3)]
    mean_guided = boxfilter(guided, radius) / base
    means_ig = [boxfilter(image[:, :, i] * guided, radius) / base for i in range(3)]
    covariance = [means_ig[i] - means[i] * mean_guided for i in range(3)]

    variance = defaultdict(dict)
    for left, right in combinations_with_replacement(range(3), 2):
        variance[left][right] = (
            boxfilter(image[:, :, left] * image[:, :, right], radius) / base
            - means[left] * means[right]
        )

    a = np.zeros((height, width, 3))
    for y, x in np.ndindex(height, width):
        sigma = np.array(
            [
                [variance[R][R][y, x], variance[R][G][y, x], variance[R][B][y, x]],
                [variance[R][G][y, x], variance[G][G][y, x], variance[G][B][y, x]],
                [variance[R][B][y, x], variance[G][B][y, x], variance[B][B][y, x]],
            ]
        )
        cov = np.array([channel[y, x] for channel in covariance])
        a[y, x] = np.dot(cov, inv(sigma + eps * np.eye(3)))

    b = mean_guided - a[:, :, R] * means[R] - a[:, :, G] * means[G] - a[:, :, B] * means[B]
    return (
        boxfilter(a[:, :, R], radius) * image[:, :, R]
        + boxfilter(a[:, :, G], radius) * image[:, :, G]
        + boxfilter(a[:, :, B], radius) * image[:, :, B]
        + boxfilter(b, radius)
    ) / base
