import torch
import torch.nn as nn
from src.models.layers import ImageToPatches3D, PatchEmbedding3D, SelfAttentionEncoderBlock, OutputProjection

class ViT_MLP(nn.Module):
    """
    Vision Transformer (ViT) with an additional Multi-Layer Perceptron (MLP) for enhanced feature extraction.
    
    This model extends the standard ViT by adding a deep MLP after the self-attention blocks to capture
    more complex patterns in the data.

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
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        # Image to patches and embedding layers
        self.i2p3d = ImageToPatches3D(image_size, patch_size)
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))

        # Create self-attention blocks for the transformer encoder
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        # New MLP layer for enhanced feature extraction
        self.new_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 32),  # Expand to 32x embed_size
            nn.GELU(),  # GELU activation for non-linearity
            nn.Linear(embed_size * 32, embed_size * 64),  # Expand to 64x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 64, embed_size * 128),  # Expand to 128x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 128, embed_size * 256),  # Expand to 256x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 256, embed_size * 512),  # Expand to 512x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 512, embed_size * 1024),  # Expand to 1024x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 1024, embed_size * 512),  # Reduce to 512x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 512, embed_size * 256),  # Reduce to 256x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 256, embed_size * 128),  # Reduce to 128x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 128, embed_size * 64),  # Reduce to 64x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 64, embed_size * 32),  # Reduce to 32x embed_size
            nn.GELU(),
            nn.Linear(embed_size * 32, embed_size),  # Back to embed_size
            nn.Dropout(p=dropout),  # Dropout for regularization
        )

        # Output projection layer to map embeddings back to the image space
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Additional output projection for visualization (after positional embedding)
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, visualize=False):
        """
        Forward pass for the ViT_MLP model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            visualize (bool): Whether to store activations for visualization.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
            list: List of activations for visualization.
        """
        activations = []  # To store activations for visualization

        # Step 1: Image to patches
        activations.append(x.clone().detach())  # Store input activation
        x = self.i2p3d(x)

        # Step 2: Apply patch embedding
        x = self.pe(x)

        if visualize:
            # Store activation after patch embedding
            vis_output = self.vis_output_proj(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        # Step 3: Add positional embeddings
        x = x + self.position_embed

        # Step 4: Apply self-attention blocks
        for head in self.attention_blocks:
            x = head(x)

        # Step 5: Apply new MLP for enhanced feature extraction
        x = self.new_mlp(x)

        # Step 6: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations


class ViT_Depth(nn.Module):
    """
    Vision Transformer (ViT) with a deeper architecture for medical image segmentation.
    
    This model uses a deeper architecture with more self-attention blocks and a larger embedding size
    to capture more complex patterns in the data.

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
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        # Image to patches and embedding layers
        self.i2p3d = ImageToPatches3D(image_size, patch_size)
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))

        # Create self-attention blocks for the transformer encoder
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        # Output projection layer to map embeddings back to the image space
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Additional output projection for visualization (after positional embedding)
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, visualize=False):
        """
        Forward pass for the ViT_Depth model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            visualize (bool): Whether to store activations for visualization.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
            list: List of activations for visualization.
        """
        activations = []  # To store activations for visualization

        # Step 1: Image to patches
        activations.append(x.clone().detach())  # Store input activation
        x = self.i2p3d(x)

        # Step 2: Apply patch embedding
        x = self.pe(x)

        if visualize:
            # Store activation after patch embedding
            vis_output = self.vis_output_proj(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        # Step 3: Add positional embeddings
        x = x + self.position_embed

        # Step 4: Apply self-attention blocks
        for head in self.attention_blocks:
            x = head(x)

        # Step 5: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations