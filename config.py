# Data configuration
BASE_DATA_DIR = './data'

CLASSES = [
    "2-CEES", "2-CEPS", "DMMP", "4-NP"
]

RAMAN_DIRS = {
    "2-CEES": "Raman_single/CEES",
    "2-CEPS": "Raman_single/CEPS",
    "DMMP": "Raman_single/DMMP",
    "4-NP": "Raman_single/4NP"
}

GC_DIRS = {
    "2-CEES": "GC_single/CEES",
    "2-CEPS": "GC_single/CEPS",
    "DMMP": "GC_single/DMMP",
    "4-NP": "GC_single/4NP"
}

# Cross-modal classes (have both Raman and GC data)
CROSS_MODAL_CLASSES = ['2-CEES', '2-CEPS', 'DMMP', '4-NP']


VAL_SPLIT = 0.2

# Training parameters
BATCH_SIZE = 512  #(4 classes × 64 samples × 2 modalities)
EPOCHS = 2000
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4  

# Model Architecture
EMBEDDING_DIM = 512  
HIDDEN_DIM = 1024   


# Data preprocessing parameters
COMMON_LENGTH = 4096  # Both Raman and GC interpolated to this length

# Raman X-axis range 
# Files will be mapped to this grid, zero-padded where no data exists
RAMAN_X_MIN = 300.0    
RAMAN_X_MAX = 3400.0  

# GC has consistent range across all classes
GC_X_MIN = 0.0910    
GC_X_MAX = 16.9960     

# Preprocessing
SAVGOL_WINDOW = 11
SAVGOL_POLY = 2
IASLS_LAM = 1e6
IASLS_P = 0.01

# Data Loading Configuration
USE_LAZY_LOADING = True 
USE_PARALLEL = True  # Use parallel preprocessing 
MAX_WORKERS = 16  # For preprocessing
NUM_WORKERS = 16  # DataLoader workers 

MODEL_DIR = 'models/ContrastiveModel_ResNet.pth'
MODEL_PLOT_DIR = 'models/ContrastiveModel_ResNet_730.pth'


# Training Hyperparameters (Single Source of Truth)

# Initial temperature (T) for the loss function
INITIAL_TEMPERATURE = 0.5

# Min/Max temperature values for clamping
MIN_TEMP = 0.01
MAX_TEMP = 1.0

# Label smoothing (ε) for regularization (0.0 = no smoothing)
LABEL_SMOOTHING = 0.1

# Gradient clipping norm to prevent exploding gradients
GRADIENT_CLIP_NORM = 1.0

