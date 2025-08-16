import torch
import torch.nn as nn
from src.models.layers import ImageToPatches3D, PatchEmbedding3D, SelfAttentionEncoderBlock, OutputProjection
from src.llm.llama import LlamaTransformer
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedVisionLlama(nn.Module):
    """
    Vision Transformer with Llama and LoRA integration for medical image segmentation.
    
    This model integrates a pre-trained Llama model with a Vision Transformer for segmentation tasks.
    LoRA (Low-Rank Adaptation) is used to fine-tune the Llama model efficiently.

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
        self.i2p3d = ImageToPatches3D(image_size, patch_size)  # Convert 3D image to patches
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)  # Embed patches
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))  # Positional embeddings

        # Llama configuration
        llm_default_config = {"dim": 4096, "multiple_of": 256,
                              "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-5,
                              "vocab_size": -1, "first_layer": 31, "kv_heads": 8}
        self.llm = LlamaTransformer(llm_default_config)

        # LoRA configuration
        lora_rank = 4  # Adjust rank as needed

        # Load Llama checkpoint
        llm_path = "/path/to/your/project/LLaMA_v3.1/llama-3.1-8b"
        logger.info("Loading Llama checkpoints")
        start_time = time.time()
        checkpoints = sorted(Path(llm_path).glob("*.pth"))
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.llm.custom_load_state_dict(checkpoint, tail=True, strict=False)
        logger.info(f"Loaded in {time.time() - start_time:.2f} seconds"

        # Freeze all parameters except LoRA layers
        lora.mark_only_lora_as_trainable(self.llm)

        # Dimensionality mapping layers
        self.llm_dim_mapper1 = lora.Linear(embed_size, 4096, r=lora_rank)  # Map ViT embeddings to Llama input size
        self.llm_dim_mapper2 = lora.Linear(4096, embed_size, r=lora_rank)  # Map Llama embeddings back to ViT size

        # Self-attention blocks for the transformer encoder
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        
        # Output projection layer to map embeddings back to the image space
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Visualization layers
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)
        self.vis_output_proj_ll1 = OutputProjection(image_size, patch_size, 4096, out_channels)
        self.vis_output_proj_llm = OutputProjection(image_size, patch_size, 4096, out_channels)
        self.vis_output_proj_ll2 = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, visualize=False):
        """
        Forward pass for the MedVisionLlama model.
        
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

        # Step 3: Map embeddings to Llama input size
        x = self.llm_dim_mapper1(x)
        if visualize:
            vis_output = self.vis_output_proj_ll1(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after mapping

        # Step 4: Pass through Llama model
        x = self.llm(x) + x
        if visualize:
            vis_output = self.vis_output_proj_llm(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after Llama

        # Step 5: Map embeddings back to ViT size
        x = self.llm_dim_mapper2(x)
        if visualize:
            vis_output = self.vis_output_proj_ll2(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after mapping back

        # Step 6: Add positional embeddings
        x = x + self.position_embed

        # Step 7: Apply self-attention blocks
        for head in self.attention_blocks:
            x = head(x)

        # Step 8: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations
