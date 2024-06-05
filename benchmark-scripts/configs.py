"""This module allows the configuration of the benchmark"""
# pylint: skip-file
# ##################### Basic Configurations ######################
MAX_TRIALS_LIST = [6000]  # Specify the number of tuning trials, e.g., [1000, 2000] to compile with 1000, followed by 2000 trials
BUILD_ONLY = False        # If set to true the model will only be build and not tuned
NUM_REPEATS = 1           # The number of times you want to compile the model
FLUSH_CACHES = False      # Flush CPU caches between measurements, enable whenever possible (does not work on macOS)
TAG = ""                  # Tag applied to the directory name, besides strategy and trials
NUM_TARGET_CORES = 16     # The number of cores we tune for and use for the runtime

# ################### Configure CPU Target ####################
# TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=skylake  -mattr=avx2"                   # Department CPU
# TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=skylake-avx512"                         # HPC
TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=apple-latest -mtriple=arm64-apple-macos"  # M3 Max Target

# ################### Configure GPU Target ####################
# TARGET_NAME = "nvidia/nvidia-a100"
# TARGET_NAME = "nvidia/tesla-p100"     
# TARGET_NAME = "apple/m2-gpu"          # Using Metal

# ################# Configure Search Strategy #################
SEARCH_STRATEGY = "bayesian"
# SEARCH_STRATEGY = "evolutionary"

# ################# Configure Tuning Cores #################
# TUNING_CORES = 1
TUNING_CORES = 16
# TUNING_CORES = 38

# ################# Select Model #################
MODEL_NAME = "mobilenet"
# MODEL_NAME = "matmul"
# MODEL_NAME = "resnet50"
# MODEL_NAME = "multilayer_perceptron"
# MODEL_NAME = "bert"
# MODEL_NAME = "resnet50-torch"
# MODEL_NAME = "mobilenetv2"
# MODEL_NAME = "gpt2"

# ############# Bayesian Optimization Settings ##############
LOG_LIMIT = 250               # The maximum numbers of observations the optimizer uses
ACQUISITION_FUNCTION = "ucb"  # Options "ucb" (Upper Confidence Bound), "poi" (Probability of Improvement), and "ei" (Expected Improvement)
KAPPA = 0.1                   # Kappa setting for UCB (the lower the more exploitation)
XI = 0.1                      # XI setting for EI and POI (Exploit: (0.0 for EI) and (1e-4 for POI); Explore (0.1 for EI and POI))

USE_SEQUENTIAL_DOMAIN_REDUCTION = False # See more here: https://doi.org/10.1108/02644400210430190
USE_MIN_HEAP = False                    # Use a heap to select the best discovered programs
USE_LRU = False                         # Use a LRU strategy to maintain the memory limit

REGISTER_FAILURE_POINTS = True          # Register invalid schedule parameters with the optimizer
