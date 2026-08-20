import numpy as np


CALIB_KNOTS = np.array([350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 720.0])
RIDGE = 1e-3
CALIB_G = None


def calib_design(x):
    x = np.asarray(x, np.float64)
    return np.column_stack([np.ones_like(x), x] +
                           [np.maximum(x - k, 0.0) for k in CALIB_KNOTS])


def calib_fit(x, y, ridge=RIDGE):
    a = calib_design(np.ravel(x))
    m = a.T @ a
    m += ridge * np.eye(a.shape[1]) * np.trace(m) / a.shape[1]
    return np.linalg.solve(m, a.T @ np.ravel(y))


def calib_apply(g):
    c = CALIB_G
    if c is None:
        return g
    return (calib_design(np.asarray(g, np.float64).ravel()) @ c).reshape(np.shape(g))


def main():
    x = np.linspace(300.0, 760.0, 128)
    y = 1.5 + 0.8 * x - 0.002 * np.maximum(x - 500.0, 0.0)
    fitted = calib_fit(x, y)
    assert fitted.shape == (10,)
    assert np.isfinite(fitted).all()

    g = np.arange(24.0).reshape(2, 12) + 300.0
    assert np.array_equal(calib_apply(g), g)

    global CALIB_G
    CALIB_G = np.array([0.0, 1.0] + [0.0] * len(CALIB_KNOTS))
    assert np.allclose(calib_apply(g), g)

    CALIB_G = np.array([12.0, 0.37, -0.02, 0.05, -0.03, 0.01, 0.02, -0.01, 0.04, -0.02])
    f = lambda z: calib_apply(np.asarray(z, np.float64))
    for left, right in zip((300.0, *CALIB_KNOTS), (*CALIB_KNOTS, 760.0)):
        mid = (left + right) / 2.0
        assert np.isclose(f(mid), (f(left) + f(right)) / 2.0)
    for knot in CALIB_KNOTS:
        assert np.isclose(f(knot - 1e-7), f(knot + 1e-7), atol=1e-7)

    out = calib_apply(g)
    assert out.shape == (2, 12)
    print("calib_apply selftest passed")


if __name__ == "__main__":
    main()
