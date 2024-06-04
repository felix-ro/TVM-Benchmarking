# TVM-Benchmarking

This repository enables the benchmarking of models compiled with Apache TVM, using RelayIR as a frontend and MetaSchedule as a scheduling system.

The benchmarking scripts were developed to work with [TVM-Bayesian-Optimization](https://github.com/felix-ro/TVM-Bayesian-Optimization), a fork of TVM that offers support for Bayesian Optimization as a search strategy. 

## Dependencies
- TVM (if you want to use Bayesian Optimization as a search strategy, clone and build [this repo](https://github.com/felix-ro/TVM-Bayesian-Optimization))
- Python 3.10
- PyTorch
- TorchVision 
- Transformers

## Usage
The easiest approach is to configure the `benchmark-scripts\configs.py` file.

Alternatively, you can use command line arguments. However, make sure to first configure the target settings in `benchmark-scripts\configs.py`.

### Possible Arguments
```
$ python benchmark-settings\main.py [--model=MODEL] [--numTrials=NUM_TRIALS] \
 [--strategy=STRATEGY] [--tag=TAG] [--acqFunc=ACQ_FUNC] [--kappa=KAPPA] \
 [--xi=XI] [--logLimit=LOG_LIMIT] [--SDR] [--LRU] [--Heap]
```

### Example 1
Compiling ResNet-50 with Evolutionary Search and 6000 trials.
```
$ python benchmark-settings\main.py --model="resnet50" --numTrials=6000 
    --strategy="evolutionary"
```

### Example 2
Compiling ResNet-50 with Bayesian Optimization with Upper Confidence Bound as the Acquisition Function, an exploration-exploitation trade-off value of 0.1, and heap as the selection policy (for more on these settings see [this](https://github.com/felix-ro/TVM-Bayesian-Optimization)). 
```
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

