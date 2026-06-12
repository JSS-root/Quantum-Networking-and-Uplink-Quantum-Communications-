import numpy as np

import uplink_qkd


def test_public_imports():
    assert hasattr(uplink_qkd, "smart_optimise")
    assert hasattr(uplink_qkd, "secure_key_rates")


def test_finite_key_objective_returns_finite_or_nan():
    value = uplink_qkd.objective(
        [0.1, 0.02, 0.01],
        delta=0.01,
        m=1_000,
        eps_qkd=1e-6,
        t=np.log2(10**8),
        f=1.19,
    )
    assert np.isscalar(value)
