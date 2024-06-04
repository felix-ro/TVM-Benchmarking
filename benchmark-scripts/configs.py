"""This module allows the configuration of the benchmark"""
# pylint: skip-file
# ##################### Configure Trials ######################
MAX_TRIALS_LIST = [6000]  # Specify the number of tuning trials, e.g., [1000, 2000] to compile with 1000, followed by 2000 trials
BUILD_ONLY = False
NUM_REPEATS = 1  # The number of times you want to compile the model
FLUSH_CACHES = False  # Flush CPU caches. Does not work on MacOS (enable whenever possible)

# ################### Configure CPU Target ####################
NUM_TARGET_CORES = 16
# TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=skylake  -mattr=avx2"  # Department CPU
# TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=skylake-avx512" #HPC
TARGET_NAME = f"llvm -num-cores {NUM_TARGET_CORES} -mcpu=apple-latest -mtriple=arm64-apple-macos"  # M3 Max Target

# ################### Configure GPU Target ####################
# TARGET_NAME = "nvidia/nvidia-a100"
# TARGET_NAME = "nvidia/tesla-p100"
# TARGET_NAME = "apple/m2-gpu"  # Using Metal

# ################# Configure Search Strategy #################
SEARCH_STRATEGY = "bayesian"
# SEARCH_STRATEGY = "evolutionary"

# ################# Configure Tuning Cores #################
# TUNING_CORES = 1
TUNING_CORES = 16
# TUNING_CORES = 38

# ################# Select Model #################
# MODEL_NAME = "mobilenet"
MODEL_NAME = "matmul"
# MODEL_NAME = "resnet50"
# MODEL_NAME = "multilayer_perceptron"
# MODEL_NAME = "bert"
# MODEL_NAME = "resnet50-torch"
# MODEL_NAME = "mobilenetv2"
# MODEL_NAME = "gpt2"

# ############# Additional Settings ##############
TAG = ""

# ### The below settings only matter if you use the bayesian strategy ###
LOG_LIMIT = 250
RESTRICTED_MEMORY_LOGGING = False

# Options "ucb" (Upper Confidence Bound), "poi" (Probability of Improvement), and "ei" (Expected Improvement)
ACQUISITION_FUNCTION = "ucb"
# Kappa setting for UCB
KAPPA = 0.1
# XI setting for EI and POI (Exploit: (0.0 for EI) and (1e-4 for POI); Explore (0.1 for EI and POI))
XI = 0.1

# See more here: https://doi.org/10.1108/02644400210430190
USE_SEQUENTIAL_DOMAIN_REDUCTION = False
REGISTER_FAILURE_POINTS = True
USE_MIN_HEAP = False
