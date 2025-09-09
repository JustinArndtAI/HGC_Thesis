import torch
import torch.nn as nn
import torch.fft as fft

class HolographicKnowledgeManifold(nn.Module):
    """
    A PyTorch module implementing a Holographic Knowledge Manifold (HKM)
    using Vector Symbolic Architectures (VSA) principles.

    This class handles the core holographic operations (binding, bundling)
    in a batch-friendly, GPU-accelerated manner.
    """
    def __init__(self, d: int):
        """
        Initializes the Holographic Knowledge Manifold.

        Args:
            d (int): The dimensionality of the holographic vectors. A high
                     dimension (e.g., 4096, 8192) is recommended.
        """
        super().__init__()
        if d % 2 != 0:
            raise ValueError("Dimensionality 'd' must be an even number for FFT.")
        self.d = d
        # The manifold itself, initialized as a zero vector.
        # This will be populated with knowledge via superposition (bundling).
        self.register_buffer('manifold', torch.zeros(d))

    def _get_device(self):
        """Helper to determine the device of the manifold tensor."""
        return self.manifold.device

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Performs batch-wise binding (circular convolution) of two tensors.

        Args:
            a (torch.Tensor): A batch of vectors of shape (batch_size, d).
            b (torch.Tensor): A batch of vectors of shape (batch_size, d).

        Returns:
            torch.Tensor: The bound tensor of shape (batch_size, d).
        """
        if a.shape != b.shape:
            raise ValueError("Input tensors must have the same shape for binding.")
        # The core of circular convolution is multiplication in the Fourier domain.
        a_fft = fft.fft(a, dim=-1)
        b_fft = fft.fft(b, dim=-1)
        result_fft = a_fft * b_fft
        # Return to the spatial domain, taking only the real part.
        return fft.ifft(result_fft, dim=-1).real

    @staticmethod
    def unbind(a_b_bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Performs batch-wise unbinding (circular correlation) to retrieve b.
        This is the approximate inverse of the bind operation.

        Args:
            a_b_bound (torch.Tensor): The bound tensor of shape (batch_size, d).
            a (torch.Tensor): The key tensor of shape (batch_size, d).

        Returns:
            torch.Tensor: The retrieved tensor, an approximation of b.
        """
        if a_b_bound.shape != a.shape:
            raise ValueError("Input tensors must have the same shape for unbinding.")
        # Circular correlation is achieved by multiplying with the complex conjugate
        # in the Fourier domain.
        a_b_fft = fft.fft(a_b_bound, dim=-1)
        a_fft = fft.fft(a, dim=-1)
        a_fft_conj = torch.conj(a_fft)
        result_fft = a_b_fft * a_fft_conj
        return fft.ifft(result_fft, dim=-1).real

    @staticmethod
    def bundle(tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Performs bundling (element-wise addition/superposition) on a list of tensors.

        Args:
            tensors (list[torch.Tensor]): A list of tensors, each of shape (d,).

        Returns:
            torch.Tensor: The bundled tensor of shape (d,).
        """
        if not tensors:
            raise ValueError("Input tensor list cannot be empty for bundling.")
        return torch.sum(torch.stack(tensors), dim=0)

    def add_to_manifold(self, knowledge_vectors: torch.Tensor):
        """
        Adds a batch of new knowledge vectors to the manifold via bundling.

        Args:
            knowledge_vectors (torch.Tensor): A tensor of shape (batch_size, d)
                                              representing new facts or concepts.
        """
        with torch.no_grad():
            # Sum the new vectors and add them to the existing manifold.
            new_knowledge = torch.sum(knowledge_vectors, dim=0)
            self.manifold += new_knowledge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass here can be used to check similarity of an input
        vector against the entire manifold.

        Args:
            x (torch.Tensor): A batch of query vectors of shape (batch_size, d).

        Returns:
            torch.Tensor: A batch of cosine similarity scores.
        """
        # Normalize both the input vectors and the manifold to compute cosine similarity
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        manifold_norm = torch.nn.functional.normalize(self.manifold.unsqueeze(0), p=2, dim=-1)
        
        # Compute cosine similarity
        return (x_norm * manifold_norm).sum(dim=-1)