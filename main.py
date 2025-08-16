# main.py
import argparse
from scripts.train import train_model
from src.utils.utils import TupleAction

if __name__ == "__main__":
    # Argument parser for command-line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing data")
    parser.add_argument('--data_file', type=str, required=True, help="CSV file with data paths")
    parser.add_argument('--task_name', type=str, required=True, help="Name of the task for saving results")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training and validation")
    parser.add_argument('--image_size', type=str, required=True, action=TupleAction, help='Image size in the format "16,16,4"')
    parser.add_argument('--patch_size', type=str, required=True, action=TupleAction, help='Patch size in the format "16,16,4"')
    parser.add_argument('--dataset_fraction', type=str, choices=['full', 'ten', 'thirty'], default='full', 
                        help="Fraction of dataset to use for training (full, ten, thirty)")
    parser.add_argument('--model_label', type=str, required=False, 
                        choices=["ViT_Baseline", "ViT_MLP", "ViT_Depth", "MedVisionLlama", 
                                 "MedVisionLlama_Frozen", "MedVision_BioGPT", "MedVision_BioBERT", "MedVision_ClinicalBERT"], 
                        help="Label of the model to be used")

    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_model(
        data_dir=args.data_dir,
        data_file=args.data_file,
        save_path=args.task_name,
        task_name=args.task_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patch_size=args.patch_size, 
        dataset_fraction=args.dataset_fraction,
        model_label=args.model_label
    )

# python main.py --data_dir /path/to/data --data_file /path/to/data_file.txt --task_name TaskXX --epochs 100 --batch_size 4 --image_size 128,128,128 --patch_size 8,8,8 --lr 2e-3 --dataset_fraction full --model_label MedVisionLlama


