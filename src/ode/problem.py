import torch


class InitialValueProblem:
    """Initial value problem for modal systems."""
    def __init__(
            self,
            fs: int,
            num_samples: int,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            mu: torch.Tensor,
            Phi_e: torch.Tensor,
            fe_points: torch.Tensor
        ):
        """Parameters
        ----------
        fs : int
            Sampling rate [Hz].
        num_samples : int
            Number of samples (i.e., duration).
        y0 : torch.Tensor
            Initial conditions (batch size, state dimension).
        omega : torch.Tensor
            Modal angular frequencies (batch size, number of modes).
        sigma : torch.Tensor
            Modal damping parameters (batch size, number of modes).
        mu : torch.Tensor
            Scaling factors of the nonlinearity (batch size,). Denoted as "nu" in the JAES paper.
        Phi_e : torch.Tensor
            Modal shapes at the excitation position (batch size, number of modes).
        fe_points : torch.Tensor
            Excitation samples (batch size, number of samples).
        """
        self.fs = fs
        self.num_samples = num_samples
        self.y0 = y0
        self.omega = omega
        self.sigma = sigma
        self.mu = mu
        self.Phi_e = Phi_e
        self.fe_points = fe_points

        if not torch.jit.is_scripting():
            assert y0.ndim == 2
            assert omega.ndim == 2
            assert sigma.ndim == 2
            assert mu.ndim == 1
            assert Phi_e.ndim == 2
            assert fe_points.ndim == 2

            num_modes = omega.shape[-1]
            assert sigma.shape[-1] == num_modes
            assert Phi_e.shape[-1] == num_modes
            assert fe_points.shape[-1] == num_samples
