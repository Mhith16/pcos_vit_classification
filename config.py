import torch

class Config:
    # Data paths
    DATA_PATH = "/content/dataset"  # Path to your dataset
    
    # Model settings
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    
    # Training settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EARLY_STOPPING_PATIENCE = 10
    
    # Model types to train
    MODELS_TO_TRAIN = [
        'vanilla_vit',
        'pretrained_vit', 
        'ensemble_vit'
    ]
    
    # Save paths
    MODEL_SAVE_PATH = "/content/drive/MyDrive/pcos_models"
    RESULTS_SAVE_PATH = "/content/drive/MyDrive/pcos_results"