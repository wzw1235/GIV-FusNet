import torch
import torchsparse


def dense_to_sparse(tensor: torch.Tensor) -> torchsparse.tensor.SparseTensor:
    """
    tensor shape: [B, C, D, H, W]
    """
    non_zero_indices = torch.nonzero(
        (tensor != 0).any(dim=1),
        as_tuple=False,
    )

    values = tensor[
        non_zero_indices[:, 0],
        :,
        non_zero_indices[:, 1],
        non_zero_indices[:, 2],
        non_zero_indices[:, 3],
    ]

    return torchsparse.SparseTensor(
        feats=values,
        coords=non_zero_indices.to(dtype=torch.int32),
    )