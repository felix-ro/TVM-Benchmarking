# TVM-Benchmarking

This repository enables the compiling and benchmarking of models with Apache TVM. The current setup uses RelayIR as a frontend and MetaSchedule as a scheduling system.

The benchmarking scripts were developed to work with [TVM](https://github.com/apache/tvm) and [TVM-Bayesian-Optimization](https://github.com/felix-ro/TVM-Bayesian-Optimization), which is a fork of TVM that offers support for Bayesian Optimization as a search strategy. 

## Table of Contents
1. [Dependencies](#dependencies)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Configuration](#configuration)
   - [Command Line Arguments](#command-line-arguments)
   - [Examples](#examples)
4. [Arguments](#arguments)
5. [Output](#output)

## Dependencies
- [TVM](https://github.com/apache/tvm) (Use [this fork](https://github.com/felix-ro/TVM-Bayesian-Optimization) for Bayesian Optimization)
- Python 3.10
- PyTorch
- TorchVision 
- Transformers

## Installation

Follow [this installation guide](https://llm.mlc.ai/docs/install/tvm.html) or these basic steps to install the dependencies:

1. Clone and build TVM:
    ```sh
    $ git clone --recursive https://github.com/apache/tvm
    $ cd tvm
    $ mkdir build
    $ cp cmake/config.cmake build/
    $ cd build
    # Now configure the `config.cmake` file in the build directory
    $ cmake ..
    $ make -j16
    ```
2. (Optional) For Bayesian Optimization, clone and build the fork
    ```sh
    $ git clone --recursive https://github.com/felix-ro/TVM-Bayesian-Optimization
    $ cd TVM-Bayesian-Optimization
    $ mkdir build
    $ cp cmake/config.cmake build/
    $ cd build
    # Now configure the `config.cmake` file in the build directory
    $ cmake ..
    $ make -j4
    ```
3. Standard TVM Python dependencies
    ```sh 
    $ pip install numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle
    ```
4. (Optional) Bayesian Optimization Python dependencies
    ```sh 
    $ pip install bayesian-optimization
    ```
5. Python Benchmark Dependencies
    ```sh 
    # Check for hardware-dependent install commands
    $ pip install torch torchvision transformers
    ```
6. Add TVM to python path
    ```sh
    $ export PYTHONPATH=/path-to-tvm-unity/python:$PYTHONPATH
    ```
## Usage
### Configuration
The easiest approach is to configure the `benchmark-scripts/configs.py` file.

### Command Line Arguments
Alternatively, you can use command-line arguments. However, make sure to first configure the target settings in `benchmark-scripts/configs.py`.

#### Possible Arguments
```sh
$ python benchmark-scripts/main.py [--model=MODEL] [--numTrials=NUM_TRIALS] \
    [--strategy=STRATEGY] [--tag=TAG] [--acqFunc=ACQ_FUNC] [--kappa=KAPPA] \
    [--xi=XI] [--logLimit=LOG_LIMIT] [--SDR] [--LRU] [--Heap]
```

### Examples
#### Example 1
Compiling ResNet-50 with Evolutionary Search and 6000 trials.
```sh
$ python benchmark-scripts/main.py --model="resnet50" --numTrials=6000  
    --strategy="evolutionary"
```

#### Example 2
Compiling ResNet-50 with Bayesian Optimization and 6000 trials. For the additional settings, we use Upper Confidence Bound as the Acquisition Function, an exploration-exploitation trade-off value of 0.1, and heap as the selection policy (for more on these settings, [see here](https://github.com/felix-ro/TVM-Bayesian-Optimization)). 
```sh
$ python benchmark-scripts/main.py --model="resnet50" --numTrials=6000  
    --strategy="bayesian" --acqFunc="ucb" --kappa="0.1" --LogLimit=250  
    --Heap --tag="UCB-Kappa-01-Log-250-Heap"
```

### Arguments
All arguments with type and description.
| Argument | Type | Description |
|----------|--------|--------|
| --model= | string | Pick a model to compile. We currently offer: "resnet50", "multilayer_perceptron", "bert", "resnet50-torch", "mobilenetv2", "gpt2"|
| --numTrials= | int | The number of trials (empirical measurements) to use during the search|
| --strategy= | string | Pick the search strategy to use, e.g., "evolutionary" and "bayesian" |
| --tag= | string | Give the working directory an additional naming attribute | 
| | | Additional settings when using Bayesian Optimization as a search strategy | 
| --acqFunc= | string | Pick the Acquisition Function. We currently offer: "ucb", "poi", and "ei" |
| --kappa= | float | Exploitation-Exploration trade-off factor for UCB. Smaller values increase exploitation |
| --xi= | float | Exploitation-Exploration trade-off factor for POI and EI. Smaller values increase exploitation |
| --logLimit= | int | Maximum number of observations to keep in memory | 
| --SDR | None | Set to use a Sequential Domain Reduction strategy |
| --LRU | None | Use a Least Recently Used strategy to maintain the memory limit |
| --Heap | None | Use a Heap to select the best-discovered programs for hardware measurement |

## Output
Every benchmark will generate the following output: 
- TVM Logs
- Database file containing the search history
- Library file
- Profiling report of the search strategy
- Profiling of the created computational graph
- Benchmarking results as CSV file (Note: benchmarking files are shared between the same model and search strategy)