import torch
import math

class SSMLayer(torch.nn.Module):
    """
    Simple layer that does SSMing. Assumes single input, single output. 
    Could be made multi-dimensional either by stacking and decorrelating,
    or by playing with the code to allow for multi input, multioutput. Should be relatively easy, 
    but need to carefully think a little about convolution of multi dim inputs.
    """
    def __init__(
        self,
        latent_dim,
        L_max,
        dt_min = 0.001,
        dt_max = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.A, self.B, self.C, self.D = self.random_SSM(latent_dim)
        self.Abar, self.Bbar, self.Cbar, self.Dbar = self.discretize(self.A, self.B, self.C, self.D, self.dt)
        self.dt = self.log_step_initializer(dt_min, dt_max)

    def random_SSM(
        self, 
        N : int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        initializing SSM parameters given latent dim
        
        parameters:
            N : size of latent dimension
        """
        A = torch.autograd.Variable(torch.rand(size=(N,N)), requires_grad = True)
        B = torch.autograd.Variable(torch.rand(size=(N,1)), requires_grad = True)
        C = torch.autograd.Variable(torch.rand(size=(1,N)), requires_grad = True)
        D = torch.autograd.Variable(torch.rand(size=(1,1)), requires_grad = True)
        return A, B, C, D

    def log_step_initializer(self, dt_min = 0.001, dt_max = 0.1):
        """
        initial guess for dt, from random number generator. to be learned.
    
        parameters:
            dt_min
            dt_max
        """
        return torch.autograd.Variable(torch.rand(1) * (torch.log(dt_max) - torch.log(dt_min)) + torch.log(dt_min), requires_grad = True)

    def discretize(
        self, A : torch.Tensor, B : torch.Tensor, C : torch.Tensor, D : torch.Tensor, delta : torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """Discretizes SSM using bilinear model
    
        parameters:
            A: (NxN) transition matrix in latent
            B: (Nx1) projection matrix to latent
            C: (1xN) projection matrix from latent to output
            D: (1x1) skip connection from input to output
            delta: time step, ensure sufficient smallness
        """
        Cbar = C
        Dbar = D
        N = A.shape[0]
        Bl = torch.linalg.inv(torch.eye(N) - delta / 2 * A)
        Abar = Bl@(torch.eye(N) + delta/2 * A)
        Bbar = Bl@(delta*B)
        return Abar, Bbar, Cbar, Dbar

    def scan_SSM(
        self, Ab : torch.Tensor, Bb : torch.Tensor, Cb : torch.Tensor, Db : torch.Tensor,  u : torch.Tensor, x0 : torch.Tensor
    ) -> torch.Tensor:
        """
        computes steps of the SSM going forward.
    
        parameters:
            Ab : (NxN) transition matrix in discrete space of latent to latent
            Bb : (Nx1) projcetion matrix from input to latent space
            Cb : (1xN) projection matrix from latent to output
            Db : (1x1) skip connection input to output
            u  : (L,)  trajectory we are trying to track
            x0 : (Nx1) initial condition of latent
        """
        x0 = torch.zeros((10,1))
        x = torch.zeros((Ab.shape[0], len(u[:100])))
        y = torch.zeros_like(u[:100])
        for i in range(u[:100].shape[0]):
            x[:,i] = (Ab@x0 + Bb*u[i]).squeeze()
            y[i] = (Cb@x[:,i]).squeeze()
            x0 = x[:,i].unsqueeze(-1)
        return x, y
        
    def K_conv(self, Ab : torch.Tensor, Bb : torch.Tensor, Cb : torch.Tensor, L : int) -> torch.Tensor:
        """
        computes convolution window given L time steps using equation K_t = Cb @ (Ab^t) @ Bb. 
        Needs to be flipped for correct causal convolution, but can be used as is in fft mode
    
        parameters:
            Ab : transition matrix
            Bb : projection matrix from input to latent
            Cb : projection matrix from latent to input
            Db : skip connection
            L  : length over which we want convolutional window
        """
        return torch.stack([(Cb @ torch.matrix_power(Ab, l) @ Bb).squeeze() for l in range(L)])

    def causal_conv(u : torch.Tensor, K : torch.Tensor, notfft : bool = False) -> torch.Tensor:
        """
        computes 1-d causal convolution either using standard method or fft transform.
    
        parameters:
            u : trajectory to convolve
            K : convolutional filter
            notfft: boolean, for whether or not we use fft mode or not.
        """
        assert K.shape==u.shape
        
        L = u.shape[0]
        powers_of_2 = 2**int(math.ceil(math.log2(2*L)))
    
        if notfft:
            padded_u = torch.nn.functional.pad(u, (L-1,L-1))
            convolve = torch.zeros_like(u)
            for i in range(L):
                convolve[i] = torch.sum(padded_u[i:i+L]*K.flip(dims=[0]))
            return convolve
        else:
    
            K_pad = torch.nn.functional.pad(K, (0, L))
            u_pad = torch.nn.functional.pad(u, (0, L))
            
            K_f, u_f = torch.fft.rfft(K_pad, n = powers_of_2), torch.fft.rfft(u_pad, n = powers_of_2)
            return torch.fft.irfft(K_f * u_f, n = powers_of_2)[:L]

    def forward(
        self,
        u : torch.Tensor,
        x0 : torch.Tensor = torch.zeros((1,1)),
        mode : bool | str = False
    ) -> torch.Tensor:
        """
        forward pass of model

        Parameters:
            u  : input time series
            x0 : initial condition, only used in recurrent mode
            mode: recurrent mode ("recurrent"), or convolution mode (True : direct convolution, False : fourier transform)
        """
        if mode == "recurrent":
            return self.scan_SSM(self.Abar, self.Bbar, self.Cbar, u, x0)[1]
        else:
            K = self.K_conv(self.Abar, self.Bbar, self.Cbar, u.shape[0])
            return self.causal_conv(u, K, mode)
        