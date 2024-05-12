import tvm

from utils import tune, build, export_library, get_mod_and_params, benchmark, log_config
import configs

import os
import getopt
import sys


def parse_arguments(raw_arguments):
    long_options = ["model=", "tag=", "kappa=", "strategy=", "numTrials=", "SDR", "RML",
                    "Heap", "LogLimit=", "acqFunc=", "xi="]

    arguments, _ = getopt.getopt(raw_arguments, "", long_options)

    try:
        for currentArgument, currentValue in arguments:
            if currentArgument == "--model":
                configs.MODEL_NAME = str(currentValue)
            if currentArgument == "--tag":
                configs.TAG = str(currentValue)
            if currentArgument == "--kappa":
                configs.KAPPA = float(currentValue)
            if currentArgument == "--strategy":
                configs.SEARCH_STRATEGY = str(currentValue)
            if currentArgument == "--numTrials":
                configs.MAX_TRIALS_LIST = [int(currentValue)]
            if currentArgument == "--SDR":
                configs.USE_SEQUENTIAL_DOMAIN_REDUCTION = True
            if currentArgument == "--RML":
                configs.RESTRICTED_MEMORY_LOGGING = True
            if currentArgument == "--Heap":
                configs.USE_MIN_HEAP = True
            if currentArgument == "--LogLimit":
                configs.LOG_LIMIT = int(currentValue)
            if currentArgument == "--acqFunc":
                configs.ACQUISITION_FUNCTION = str(currentValue)
            if currentArgument == "--xi":
                configs.XI = float(currentValue)
    except getopt.error as err:
        print(str(err))


def main():
    parse_arguments(sys.argv[1:])
    os.environ["NUMEXPR_MAX_THREADS"] = str(configs.TUNING_CORES)
    os.environ["TVM_NUM_THREADS"] = str(configs.NUM_TARGET_CORES)
    if configs.LOG_LIMIT is not None:
        os.environ["TVM_BO_MAX_OPTIMIZER_ENTRIES"] = str(configs.LOG_LIMIT)
    if configs.USE_SEQUENTIAL_DOMAIN_REDUCTION is not None:
        os.environ["TVM_BO_USE_SEQUENTIAL_DOMAIN_REDUCTION"] = str(configs.USE_SEQUENTIAL_DOMAIN_REDUCTION)
    if configs.RESTRICTED_MEMORY_LOGGING is not None:
        os.environ["TVM_BO_RESTRICTED_MEMORY_LOGGING"] = str(configs.RESTRICTED_MEMORY_LOGGING)
    if configs.ACQUISITION_FUNCTION is not None:
        os.environ["TVM_BO_ACQUISITION_FUNCTION"] = configs.ACQUISITION_FUNCTION
    if configs.KAPPA is not None:
        os.environ["TVM_BO_KAPPA"] = str(configs.KAPPA)
    if configs.XI is not None:
        os.environ["TVM_BO_XI"] = str(configs.XI)
    if configs.REGISTER_FAILURE_POINTS is not None:
        os.environ["TVM_BO_REGISTER_FAILURE_POINTS"] = str(configs.REGISTER_FAILURE_POINTS)
    if configs.USE_MIN_HEAP is not None:
        os.environ["TVM_BO_USE_MIN_HEAP"] = str(configs.USE_MIN_HEAP)

    target = tvm.target.Target(configs.TARGET_NAME)
    mod, params = get_mod_and_params()

    for trials in configs.MAX_TRIALS_LIST:
        for i in range(configs.NUM_REPEATS):
            work_dir = f"results/{configs.MODEL_NAME}-{trials}-{configs.SEARCH_STRATEGY}-{configs.TAG}-{i}/"
            log_config(work_dir)
            if configs.BUILD_ONLY:
                graph_module, lib = build(mod=mod, params=params, target=target)
                export_library(lib=lib, model_name=configs.MODEL_NAME, target_name=configs.TARGET_NAME,
                               work_dir=work_dir, max_trials=trials)
                profile_results = None
            else:
                graph_module, lib, profile_results = tune(mod=mod, params=params, target=target,
                                                          work_dir=work_dir, max_trials=trials)

                export_library(lib=lib, model_name=configs.MODEL_NAME, target_name=configs.TARGET_NAME,
                               work_dir=work_dir, max_trials=trials)

            benchmark(target=target, lib=lib, work_dir=work_dir, graph_module=graph_module,
                      num_trials=trials, profile_results=profile_results)


if __name__ == "__main__":
    main()
