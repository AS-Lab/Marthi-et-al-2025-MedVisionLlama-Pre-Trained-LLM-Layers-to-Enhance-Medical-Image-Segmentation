import torch
import torch.nn as nn
from src.models.layers import ImageToPatches3D, PatchEmbedding3D, SelfAttentionEncoderBlock, OutputProjection
from src.llm.llama import LlamaTransformer
import loralib as lora
from transformers import AutoModel
import time
from pathlib import Path
import logging
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedVisionLlama_Frozen(nn.Module):
    """
    Vision Transformer (ViT) with a frozen Llama model and linear layers for medical image segmentation.
    
    This model integrates a pre-trained Llama model with a Vision Transformer for segmentation tasks.
    The Llama model is frozen, meaning its weights are not updated during training, while the rest of the
    model is trainable.

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

        # Llama configuration
        llm_default_config = {"dim": 4096, "multiple_of": 256,
                              "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-5,
                              "vocab_size": -1, "first_layer": 31, "kv_heads": 8}
        self.llm = LlamaTransformer(llm_default_config)

        # Load Llama checkpoint
        llm_path = "/path/to/your/project/LLaMA_v3.1/llama-3.1-8b"
        logger.info("Loading Llama checkpoints")
        start_time = time.time()
        checkpoints = sorted(Path(llm_path).glob("*.pth"))
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.llm.custom_load_state_dict(checkpoint, tail=True, strict=False)
        logger.info(f"Loaded in {time.time() - start_time:.2f} seconds"

        # Freeze Llama parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Dimensionality mapping layers
        self.llm_dim_mapper1 = nn.Linear(embed_size, 4096, bias=False)
        self.llm_dim_mapper2 = nn.Linear(4096, embed_size, bias=False)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        
        # Output projection layer
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
        Forward pass for the MedVisionLlama_Frozen model.
        
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

        # Step 4: Pass through frozen Llama model
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


class MedVision_BioGPT(nn.Module):
    """
    Vision Transformer (ViT) with BioGPT integration for medical image segmentation.
    
    This model integrates a pre-trained BioGPT model with a Vision Transformer for segmentation tasks.
    BioGPT is used to process text inputs (if provided) and combine them with image embeddings.

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

        # Load BioGPT model from a specified path (local or Hugging Face)
        self.biogpt = AutoModel.from_pretrained("/path/to/your/local/biogpt-model-directory") # https://huggingface.co/microsoft/biogpt/tree/main
        
        # LoRA configuration
        lora_rank = 4  # Adjust rank as needed

        # Freeze all parameters except LoRA layers
        lora.mark_only_lora_as_trainable(self.biogpt)

        # Freeze BioGPT weights
        for param in self.biogpt.parameters():
            param.requires_grad = False

        # Linear layers to map between ViT and BioGPT embeddings
        self.vit_to_biogpt = lora.Linear(embed_size, 1024, r=lora_rank)  # Map ViT embeddings to BioGPT input size
        self.biogpt_to_vit = lora.Linear(1024, embed_size, r=lora_rank)  # Map BioGPT embeddings back to ViT size

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        
        # Output projection layer
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Visualization layers
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)
        self.vis_output_proj_llm = OutputProjection(image_size, patch_size, 64, out_channels)
        self.vis_output_proj_ll2 = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, text_input_ids=None, visualize=False):
        """
        Forward pass for the MedVision_BioGPT model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            text_input_ids (torch.Tensor, optional): Input text token IDs for BioGPT.
            visualize (bool): Whether to store activations for visualization.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
            list: List of activations for visualization.
        """
        activations = []  # To store activations for visualization

        # Step 1: Image to patches
        x = self.i2p3d(x)

        # Step 2: Apply patch embedding
        x = self.pe(x)

        # Step 3: Map ViT embeddings to BioGPT input size
        x_biogpt = self.vit_to_biogpt(x)
        if visualize:
            vis_output = self.vis_output_proj_llm(x_biogpt)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after mapping

        # Step 4: Pass text input through BioGPT (if provided)
        if text_input_ids is not None:
            # Get BioGPT embeddings for the text
            biogpt_outputs = self.biogpt(input_ids=text_input_ids)
            biogpt_embeddings = biogpt_outputs.last_hidden_state  # Shape: (batch_size, seq_len, 1024)

            # Average pooling over the sequence dimension to get a fixed-size embedding
            biogpt_embeddings = biogpt_embeddings.mean(dim=1)  # Shape: (batch_size, 1024)
            biogpt_embeddings = biogpt_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, 1024)

            # Combine ViT and BioGPT embeddings (e.g., concatenation)
            x_combined = torch.cat((x_biogpt, biogpt_embeddings.expand(-1, x_biogpt.size(1), -1)), dim=-1)
            x_combined = self.biogpt_to_vit(x_combined)  # Map back to ViT size
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back
        else:
            x_combined = x_biogpt
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back

        # Step 5: Add positional embeddings
        x = x + self.position_embed

        # Step 6: Pass through self-attention blocks
        for head in self.attention_blocks:
            x = head(x_combined)

        # Step 7: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations


class MedVision_BioBERT(nn.Module):
    """
    Vision Transformer (ViT) with BioBERT integration for medical image segmentation.
    
    This model integrates a pre-trained BioBERT model with a Vision Transformer for segmentation tasks.
    BioBERT is used to process text inputs (if provided) and combine them with image embeddings.

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

        # Load BioBERT from a specified path (local or Hugging Face)
        self.biobert = AutoModel.from_pretrained("/path/to/your/local/biobert-model-directory")  # https://huggingface.co/allenai/biobert-base-cased-v1.1
        
        # LoRA configuration
        lora_rank = 4  # Adjust rank as needed

        # Freeze all parameters except LoRA layers
        lora.mark_only_lora_as_trainable(self.biobert)

        # Freeze BioBERT weights
        for param in self.biobert.parameters():
            param.requires_grad = False

        # Linear layers to map between ViT and BioBERT embeddings
        self.vit_to_biobert = lora.Linear(embed_size, 1024, r=lora_rank)  # Map ViT embeddings to BioBERT input size
        self.biobert_to_vit = lora.Linear(1024, embed_size, r=lora_rank)  # Map BioBERT embeddings back to ViT size

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        
        # Output projection layer
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Visualization layers
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)
        self.vis_output_proj_llm = OutputProjection(image_size, patch_size, 64, out_channels)
        self.vis_output_proj_ll2 = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, text_input_ids=None, visualize=False):
        """
        Forward pass for the MedVision_BioBERT model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            text_input_ids (torch.Tensor, optional): Input text token IDs for BioBERT.
            visualize (bool): Whether to store activations for visualization.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
            list: List of activations for visualization.
        """
        activations = []  # To store activations for visualization

        # Step 1: Image to patches
        x = self.i2p3d(x)

        # Step 2: Apply patch embedding
        x = self.pe(x)

        # Step 3: Map ViT embeddings to BioBERT input size
        x_biobert = self.vit_to_biobert(x)
        if visualize:
            vis_output = self.vis_output_proj_llm(x_biobert)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after mapping

        # Step 4: Pass text input through BioBERT (if provided)
        if text_input_ids is not None:
            # Get BioBERT embeddings for the text
            biobert_outputs = self.biobert(input_ids=text_input_ids)
            biobert_embeddings = biobert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, 1024)

            # Average pooling over the sequence dimension to get a fixed-size embedding
            biobert_embeddings = biobert_embeddings.mean(dim=1)  # Shape: (batch_size, 1024)
            biobert_embeddings = biobert_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, 1024)

            # Combine ViT and BioBERT embeddings (e.g., concatenation)
            x_combined = torch.cat((x_biobert, biobert_embeddings.expand(-1, x_biobert.size(1), -1)), dim=-1)
            x_combined = self.biobert_to_vit(x_combined)  # Map back to ViT size
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back
        else:
            x_combined = x_biobert
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back

        # Step 5: Add positional embeddings
        x = x + self.position_embed

        # Step 6: Pass through self-attention blocks
        for head in self.attention_blocks:
            x = head(x_combined)

        # Step 7: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations
    
class MedVision_ClinicalBERT(nn.Module):
    """
    Vision Transformer (ViT) with ClinicalBERT integration for medical image segmentation.
    
    This model integrates a pre-trained ClinicalBERT model with a Vision Transformer for segmentation tasks.
    ClinicalBERT is used to process text inputs (if provided) and combine them with image embeddings.

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

        # Load ClinicalBERT from a specified path (local or Hugging Face)
        self.clinicalbert = AutoModel.from_pretrained("/path/to/your/local/clinicalbert-model-directory")  # https://huggingface.co/emilyalsentzer/clinicalbert
        
        # LoRA configuration
        lora_rank = 4  # Adjust rank as needed

        # Freeze all parameters except LoRA layers
        lora.mark_only_lora_as_trainable(self.clinicalbert)

        # Freeze ClinicalBERT weights
        for param in self.clinicalbert.parameters():
            param.requires_grad = False

        # Linear layers to map between ViT and ClinicalBERT embeddings
        self.vit_to_clinicalbert = lora.Linear(embed_size, 1024, r=lora_rank)  # Map ViT embeddings to ClinicalBERT input size
        self.clinicalbert_to_vit = lora.Linear(1024, embed_size, r=lora_rank)  # Map ClinicalBERT embeddings back to ViT size

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        
        # Output projection layer
        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

        # Visualization layers
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)
        self.vis_output_proj_llm = OutputProjection(image_size, patch_size, 64, out_channels)
        self.vis_output_proj_ll2 = OutputProjection(image_size, patch_size, embed_size, out_channels)

    def forward(self, x, text_input_ids=None, visualize=False):
        """
        Forward pass for the MedVision_ClinicalBERT model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width, depth).
            text_input_ids (torch.Tensor, optional): Input text token IDs for ClinicalBERT.
            visualize (bool): Whether to store activations for visualization.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width, depth).
            list: List of activations for visualization.
        """
        activations = []  # To store activations for visualization

        # Step 1: Image to patches
        x = self.i2p3d(x)

        # Step 2: Apply patch embedding
        x = self.pe(x)

        # Step 3: Map ViT embeddings to ClinicalBERT input size
        x_clinicalbert = self.vit_to_clinicalbert(x)
        if visualize:
            vis_output = self.vis_output_proj_llm(x_clinicalbert)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())  # Store activation after mapping

        # Step 4: Pass text input through ClinicalBERT (if provided)
        if text_input_ids is not None:
            # Get ClinicalBERT embeddings for the text
            clinicalbert_outputs = self.clinicalbert(input_ids=text_input_ids)
            clinicalbert_embeddings = clinicalbert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, 1024)

            # Average pooling over the sequence dimension to get a fixed-size embedding
            clinicalbert_embeddings = clinicalbert_embeddings.mean(dim=1)  # Shape: (batch_size, 1024)
            clinicalbert_embeddings = clinicalbert_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, 1024)

            # Combine ViT and ClinicalBERT embeddings (e.g., concatenation)
            x_combined = torch.cat((x_clinicalbert, clinicalbert_embeddings.expand(-1, x_clinicalbert.size(1), -1)), dim=-1)
            x_combined = self.clinicalbert_to_vit(x_combined)  # Map back to ViT size
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back
        else:
            x_combined = x_clinicalbert
            if visualize:
                vis_output = self.vis_output_proj_ll2(x_combined)
                vis_output = self.sigmoid(vis_output)
                activations.append(vis_output.clone().detach())  # Store activation after mapping back

        # Step 5: Add positional embeddings
        x = x + self.position_embed

        # Step 6: Pass through self-attention blocks
        for head in self.attention_blocks:
            x = head(x_combined)

        # Step 7: Output projection and sigmoid activation
        x = self.output_proj(x)
        activations.append(x.clone().detach())  # Store activation after output projection

        x = self.sigmoid(x)
        activations.append(x.clone().detach())  # Store final activation

        return x, activations
