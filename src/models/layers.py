import torch
import torch.nn as nn
import dataclasses
import unfoldNd

class ImageToPatches3D(nn.Module):
    """
    Converts a 3D image into patches for Vision Transformer (ViT) processing.
    
    Args:
        image_size (tuple): Size of the input image in the format (height, width, depth).
        patch_size (tuple): Size of each patch in the format (height, width, depth).
    """
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = unfoldNd.UnfoldNd(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass to convert 3D image into patches.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
        
        Returns:
            torch.Tensor: Patches of shape (batch_size, num_patches, patch_size * channels).
        """
        assert len(x.size()) == 5
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.permute(0, 2, 1)
        return x_unfolded


class PatchEmbedding3D(nn.Module):
    """
    Embeds 3D patches into a lower-dimensional space.
    
    Args:
        in_channels (int): Number of input channels (patch_size * channels).
        embed_size (int): Size of the embedding dimension.
    """
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        """
        Forward pass to embed patches.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, in_channels).
        
        Returns:
            torch.Tensor: Embedded patches of shape (batch_size, num_patches, embed_size).
        """
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x


class MultiLayerPerceptron(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with GELU activation and dropout.
    
    Args:
        embed_size (int): Size of the input and output embeddings.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """
        Forward pass for the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_size).
        """
        return self.layers(x)


class SelfAttentionEncoderBlock(nn.Module):
    """
    A self-attention block for the Vision Transformer encoder.
    
    Args:
        embed_size (int): Size of the input and output embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        """
        Forward pass for the self-attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_size).
        """
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class OutputProjection(nn.Module):
    """
    Projects the embeddings back to the original image space.
    
    Args:
        image_size (tuple): Size of the output image in the format (height, width, depth).
        patch_size (tuple): Size of each patch in the format (height, width, depth).
        embed_size (int): Size of the input embeddings.
        out_channels (int): Number of output channels.
    """
    def __init__(self, image_size, patch_size, embed_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.projection = nn.Linear(embed_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        self.fold = unfoldNd.FoldNd(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass for the output projection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
        """
        B, T, C = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x


@dataclasses.dataclass
class ViTArgs:
    """
    Dataclass for Vision Transformer arguments.
    
    Args:
        image_size (tuple): Size of the input image in the format (height, width, depth).
        patch_size (tuple): Size of each patch in the format (height, width, depth).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        embed_size (int): Size of the embedding dimension.
        num_blocks (int): Number of self-attention blocks.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (16, 16, 16)
    in_channels: int = 1
    out_channels: int = 1
    embed_size: int = 64
    num_blocks: int = 16
    num_heads: int = 4
    dropout: float = 0.2


@dataclasses.dataclass
class ViTArgs_Depth:
    """
    Dataclass for Vision Transformer arguments with deeper architecture.
    
    Args:
        image_size (tuple): Size of the input image in the format (height, width, depth).
        patch_size (tuple): Size of each patch in the format (height, width, depth).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        embed_size (int): Size of the embedding dimension.
        num_blocks (int): Number of self-attention blocks.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (16, 16, 16)
    in_channels: int = 1
    out_channels: int = 1
    embed_size: int = 1024
    num_blocks: int = 20
    num_heads: int = 32
    dropout: float = 0.1