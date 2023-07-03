import numpy as np

from .sample import Sample


def _pad_for_broadcasting(array, broadcast_with):
    """Pads `array` with singleton dimensions so that it broadcasts with
    all arrays in `broadcast_with` along first axis.
    """
    index_pad_dims = np.max([np.ndim(a) for a in broadcast_with])
    return np.asarray(array).reshape(-1, *(1,) * index_pad_dims)


def _prepare_sample(sample=None, eps_stack=None, beta_stack=None, t_stack=None):
    if sample is None:
        sample = Sample(eps_stack=eps_stack, beta_stack=beta_stack, t_stack=t_stack)
    elif any([stack is not None for stack in (eps_stack, beta_stack, t_stack)]):
        raise ValueError(
            " ".join(
                [
                    "`sample` must be set to None if any of `eps_Stack`,"
                    "`beta_stack` or `t_stack` are not None."
                ]
            )
        )
    return sample


defaults = dict(
    eps_env=1 + 0j,
    d_Q0=1.31 * 15 / 17,  # But calculate from `r_tip` and `L_tip` when possible
    d_Q1=0.5,
    r_tip=20e-9,
    L_tip=300e-9,
    g_factor=0.7 * np.exp(0.06j),
    n_trapz=64,
    n_lag=64,
    n_tayl=16,
    beta_threshold=1.01,
)
