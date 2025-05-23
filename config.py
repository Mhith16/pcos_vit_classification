import torch

class Config:
    # Data paths
    DATA_PATH = "/content/dataset"  # Path to your dataset
    
    # Model settings - OPTIMIZED FOR T4
    IMG_SIZE = 224
    BATCH_SIZE = 32  # Increased for better GPU utilization
    NUM_EPOCHS = 30  # Reduced epochs
    LEARNING_RATE = 2e-4  # Slightly higher for faster convergence
    NUM_CLASSES = 2
    
    # Training settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EARLY_STOPPING_PATIENCE = 7  # Reduced patience
    
    # Performance optimizations
    USE_MIXED_PRECISION = True
    USE_GRADIENT_CHECKPOINTING = True
    NUM_WORKERS = 4  # For faster data loading
    PIN_MEMORY = True
    
    # Model types to train - START WITH FASTEST
    MODELS_TO_TRAIN = [
        'pretrained_vit_small',  # Start with smallest model
        'pretrained_vit', 
        # 'vanilla_vit',  # Skip for now - very slow
        # 'ensemble_vit'  # Skip for now - very slow
    ]
    
    # Save paths
    MODEL_SAVE_PATH = "/content/drive/MyDrive/pcos_models"
    RESULTS_SAVE_PATH = "/content/drive/MyDrive/pcos_results"