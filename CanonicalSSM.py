import torch
import math

def log_step_initializer(self, dt_min = 0.001, dt_max = 0.1):
    """
    initial guess for dt, from random number generator. to be learned.

    parameters:
        dt_min
        dt_max
    """
    return torch.autograd.Variable(torch.rand(1) * (torch.log(dt_max) - torch.log(dt_min)) + torch.log(dt_min), requires_grad = True)

def scan_SSM(
    Ab : torch.Tensor, Bb : torch.Tensor, Cb : torch.Tensor, Db : torch.Tensor,  u : torch.Tensor, x0 : torch.Tensor
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
    return x, y + Db*u

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
        dt_min = torch.tensor(0.001),
        dt_max = torch.tensor(0.1),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.A, self.B, self.C, self.D = self.random_SSM(latent_dim)
        self.dt = log_step_initializer(dt_min, dt_max)
        self.Abar, self.Bbar, self.Cbar, self.Dbar = self.discretize(self.A, self.B, self.C, self.D, self.dt)


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
            return scan_SSM(self.Abar, self.Bbar, self.Cbar, self.Dbar, u, x0)[1]
        else:
            K = self.K_conv(self.Abar, self.Bbar, self.Cbar, u.shape[0])
            return causal_conv(u, K, mode) + self.D*u



class S4Layer(torch.nn.Module):
    """
    Efficient layer for S4Ms. (Structured State Space Sequence Models).
    Implements initialization of A as a NPLR matrix, enabling fast 
    matrix vector multiplication. 

    Several parameters, such as the projection matrix, are learned.

    In this case, the C matrix is actually learned as C(1-A^L). 
    This is fairly easy to undo, and is done in the calc of Cbar.

    Parameters:
        N_input : dimension of input,
        latent_dim : int, dimensions of latent space,
        
    """
    def __init__(
        self,
        N_input : int,
        latent_dim : int,
        dt_min  : torch.Tensor = torch.tensor(0.001),
        dt_max  : torch.Tensor = torch.tensor(0.1),
    ):
        super().__init__()
        assert N_input==1


        self.latent_dim = latent_dim
        self.Lambda, self.P, self.B, _ = self.make_DPLR_HiPPO(self.latent_dim)
        
        self.Lambda = torch.autograd.Variable(self.Lambda, requires_grad = True)
        self.P = torch.autograd.Variable(self.P, requires_grad = True)
        self.B = torch.autograd.Variable(self.B, requires_grad = True)
        
        self.logdt = log_step_initializer(dt_min, dt_max)
        
        Ctilde = torch.nn.init.normal_(torch.empty(self.latent_dim, 2), mean=0, std=0.5**0.5)
        self.Ctilde = torch.autograd.Variable(Ctilde[:,0] + Ctilde[:,1]*1j, requires_grad=True)

        self.D = torch.autograd.Variable(torch.tensor(1.), requires_grad = True)

    def make_HiPPO(self, N : int) -> torch.Tensor:
        """
        creates HiPPO matrix for legendre polynomials up to order N
        parameters:
            N: int
        """
        P = torch.sqrt(1+2*torch.arange(N))
        A = P.unsqueeze(1) * P.unsqueeze(0)
        A = torch.tril(A) - torch.diag(torch.arange(N))
        return -A
        
    def make_NPLR_HiPPO(self, N : int) -> torch.Tensor:
        """
        creating hippo matrix and associated low rank additive component, P
        and the B matrix associated, as hippo forces it
    
        parameters:
            N : int, degree of legendre polynomial coefficient
        """
        nhippo = self.make_HiPPO(N)
    
        P = torch.sqrt(torch.arange(N)+0.5).to(torch.complex64)
        B = torch.sqrt(2*torch.arange(N)+1.0).to(torch.complex64)
    
        return nhippo.to(torch.complex64), P, B

    def make_DPLR_HiPPO(self, N : int) -> torch.Tensor:
        """
        convert matrices to DPLR representation
        parameters:
            N : int, degree of legendre polynomials
        """
        A, P, B = self.make_NPLR_HiPPO(N)
    
        S = A + torch.outer(P, P)
    
        S_diag = torch.diagonal(S)
        Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)
    
        Lambda_imag, V = torch.linalg.eigh(S * -1j)
        P = V.T.conj() @ P
        B = V.T.conj() @ B
        return Lambda_real + 1j * Lambda_imag, P, B, V

    
    def K_gen_DPLR(
        self,
        Lambda : torch.Tensor, 
        P : torch.Tensor, 
        Q : torch.Tensor, 
        B: torch.Tensor, 
        C : torch.Tensor, 
        delta : torch.Tensor, 
        L : int
    )-> torch.Tensor:
        """
        computes convolution kernel from generating function using DPLR representation and
        the cauchy kernel
    
        Parameters:
            Lambda : diagonal part of DPLR
            P : N matrix, rank 1 representation to A
            Q : N matrix, rank 1 representation to A
            C : N matrix, projection from latent to input
            B : N matrix, projection from input to latent
        """
        Omega_L = torch.exp(-2j*torch.pi * (torch.arange(L))/L)
    
        aterm = (torch.conj(C), torch.conj(Q))
        bterm = (B, P)
    
        g = (2.0/delta) * ((1.0-Omega_L)/(1.0+Omega_L))
        c = 2.0 / (1.0+Omega_L)
    
        k00 = self.cauchy(aterm[0] * bterm[0].unsqueeze(0), g.unsqueeze(1), Lambda)
        k01 = self.cauchy(aterm[0] * bterm[1].unsqueeze(0), g.unsqueeze(1), Lambda)
        k10 = self.cauchy(aterm[1] * bterm[0].unsqueeze(0), g.unsqueeze(1), Lambda)
        k11 = self.cauchy(aterm[1] * bterm[1].unsqueeze(0), g.unsqueeze(1), Lambda)
    
        atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
        out = torch.fft.ifft(atRoots, L)
        return out.real

    
    def cauchy(self, k : torch.Tensor, omega : torch.Tensor, lambd : torch.Tensor):
        """
        computes cauchy kernel 
        sum(c_i * b_i/(z - lambda_i)

        Parameters:
            k : term by term dot product of vectors
            omega : function of the roots of unity
            lambd: diagonal parts of the DPLR matrix
        """
        return torch.sum(k/(omega-lambd), axis=1)

    def discrete_DPLR(
        self,
        Lambda : torch.Tensor,
        P : torch.Tensor,
        Q : torch.Tensor,
        B : torch.Tensor,
        C : torch.Tensor,
        delta : torch.Tensor,
        L : int
    )->(torch.Tensor, torch.Tensor, torch.Tensor):
        """
        computes the discretized version of the state space model,
        assuming the DPLR form
    
        Parameters:
            Lambda : Nx1, represents the diagonal values of the A matrix
            P : Nx1, represents part of the low rank aspect of the A matrix
            Q : Nx1, represents the other part of the low rank aspect of the A matrix
            B : N, projection from input to latent
            C : N, projection from latent to input
            delta : step size
            L : length of window
        """
        Bt = B.unsqueeze(1)
        Ct = C.unsqueeze(0)
    
        A = (torch.diag(Lambda) - torch.outer(P, torch.conj(Q)))
        A0 = 2.0/delta * torch.eye(A.shape[0]) + A
    
        Qdagger = torch.conj(torch.transpose(Q))
        
        D = torch.diag(1.0/(2.0/delta - Lambda))
        A1 = (D -  (1.0/(1.0 + Qdagger @ D @ P)) * D@P@Qdagger@D)
        Ab = A1@A0
        Bb = 2 * A1
        Cb = Ct @ torch.conj(torch.linalg.inv(torch.eye(A.shape[0]) - torch.matrix_power(Ab, L)))
        return Ab, Bb, Cb.conj()

    def forward(self):
        pass