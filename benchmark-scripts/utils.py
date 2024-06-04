"""This module provides benchmarking utilities"""

import os
from time import sleep
from pathlib import Path
from typing import Tuple

import tvm
from tvm.relay import testing
from tvm import relay, meta_schedule
from tvm.target.target import Target
from tvm.meta_schedule.runner import LocalRunner, EvaluatorConfig
from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.cost_model import XGBModel
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from tvm.runtime.module import BenchmarkResult

import configs


def get_mod_and_params():
    """Retrieves the IRModule and parameters for a network"""
    if configs.MODEL_NAME == "mobilenet":
        return testing.mobilenet.get_workload(batch_size=1,
                                              num_classes=1000,
                                              image_shape=(3, 224, 224),
                                              dtype='float32',
                                              layout='NCHW')
    elif configs.MODEL_NAME == "resnet50":
        return testing.resnet.get_workload(
            num_layers=50,
            batch_size=1,
            layout="NHWC",
            dtype="float32",
            image_shape=(224, 224, 3),
        )
    elif configs.MODEL_NAME == "resnet50-torch":
        # pylint: disable=import-outside-toplevel
        import torchvision
        import torch

        model = torchvision.models.resnet50()
        model.eval()

        tvm_input_shape = [1, 3, 224, 224]
        input_shape = torch.randn(tvm_input_shape, dtype=torch.float32)
        scripted_func = torch.jit.trace(model, input_shape)

        tvm_input_name = 'input_ids'
        tvm_shape = [(tvm_input_name, tvm_input_shape)]

        return relay.frontend.from_pytorch(scripted_func, tvm_shape, default_dtype="float32")
    elif configs.MODEL_NAME == "mobilenetv2":
        # pylint: disable=import-outside-toplevel
        import torchvision
        import torch

        model = torchvision.models.mobilenet_v2()
        model.eval()

        tvm_input_shape = [1, 3, 224, 224]
        input_shape = torch.randn(tvm_input_shape, dtype=torch.float32)
        scripted_func = torch.jit.trace(model, input_shape)

        tvm_input_name = 'input_ids'
        tvm_shape = [(tvm_input_name, tvm_input_shape)]

        return relay.frontend.from_pytorch(scripted_func, tvm_shape, default_dtype="float32")
    elif configs.MODEL_NAME == "matmul":
        # pylint: disable=import-outside-toplevel
        import torch

        M = 256  # pylint: disable=invalid-name
        tensor_a = torch.randn((M, M), dtype=torch.float32)
        tensor_b = torch.randn((M, M), dtype=torch.float32)
        scripted_func = torch.jit.trace(torch.matmul, (tensor_a, tensor_b))
        tensor_a_name = "tensor_a"
        tensor_b_name = "tensor_b"
        shape_list = [(tensor_a_name, tensor_a.shape), (tensor_b_name, tensor_b.shape)]
        return relay.frontend.from_pytorch(scripted_func, shape_list)
    elif configs.MODEL_NAME == "multilayer_perceptron":
        return testing.mlp.get_workload(
            num_classes=1000,
            batch_size=1,
            dtype="float32",
            image_shape=(1, 28, 28),
        )
    elif configs.MODEL_NAME == "bert":
        # pylint: disable=import-outside-toplevel
        from transformers import BertModel
        import torch

        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
        model.eval()

        input_shape = [1, 128]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)  # pylint: disable=invalid-name
        traced_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]

        mod, params = relay.frontend.from_pytorch(traced_model, shape_list, default_dtype="float32")
        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        bind_pass = tvm.relay.transform.function_pass(
                        lambda fn, new_mod, ctx:
                        tvm.relay.build_module.bind_params_by_name(fn, params),
                        opt_level=1)
        mod = bind_pass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        return mod, params
    elif configs.MODEL_NAME == "gpt2":
        # pylint: disable=import-outside-toplevel
        from transformers import GPT2Model
        import torch

        model = GPT2Model.from_pretrained("gpt2", torchscript=True)
        model.eval()

        input_shape = [1, 128]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)  # pylint: disable=invalid-name
        traced_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [(input_name, input_shape)]

        mod, params = relay.frontend.from_pytorch(traced_model, shape_list, default_dtype="float32")

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        bind_pass = tvm.relay.transform.function_pass(
                        lambda fn, new_mod, ctx:
                        tvm.relay.build_module.bind_params_by_name(fn, params),
                        opt_level=1)
        mod = bind_pass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        return mod, params
    elif configs.MODEL_NAME == "vgg16":
        return testing.vgg.get_workload(batch_size=1,
                                        num_classes=1000,
                                        image_shape=(3, 224, 224),
                                        dtype="float32",
                                        num_layers=16)
    elif configs.MODEL_NAME == "inceptionV3":
        return testing.inception_v3.get_workload(batch_size=1,
                                                 num_classes=1000,
                                                 image_shape=(3, 299, 299),
                                                 dtype="float32")


def tune(mod: tvm.IRModule, params: dict, target: Target, work_dir: str,
         max_trials: int) -> Tuple[GraphModule, ExecutorFactoryModule, meta_schedule.Profiler]:
    """Tunes a network using MetaSchedule
    
    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule of the network
    params: dict
        The parameters
    target: Target
        The target device
    workd_dir: str
        The path to the working directory in which progress is saved
    max_trials: int
        The maximum number of trials used for tuning
    """
    with meta_schedule.Profiler() as profiler:

        evaluator_config = EvaluatorConfig(number=7,
                                           repeat=1,
                                           min_repeat_ms=200,
                                           enable_cpu_cache_flush=configs.FLUSH_CACHES)

        database = meta_schedule.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=max_trials,
            strategy=configs.SEARCH_STRATEGY,
            builder=LocalBuilder(max_workers=16),
            runner=LocalRunner(evaluator_config=evaluator_config),
            num_tuning_cores=configs.TUNING_CORES,
            num_trials_per_iter=64,
            cost_model=XGBModel()
        )
        lib: ExecutorFactoryModule = meta_schedule.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
            backend='graph',
        )

    print(profiler.table())
    with open(f"{work_dir}profiler.log", "w", encoding="utf-8",) as file:
        file.write(str(profiler.table()))
    device = tvm.device(str(target), 0)
    graph_module = GraphModule(lib["default"](device))
    return graph_module, lib, profiler


def build(mod: tvm.IRModule, params: dict, target: Target):
    """Build executable"""
    with tvm.transform.PassContext(opt_level=3):
        lib: ExecutorFactoryModule = relay.build_module.build(
                                            mod,
                                            target=target,
                                            params=params
                                        )
        dev = tvm.device(str(target), 0)
        graph_module = GraphModule(lib["default"](dev))
    return graph_module, lib


def get_simplified_target_name(target_name: str) -> str:
    """Returns a simplified target name
    
    Parameters
    ----------
    target_name: str
        The long target name
    """
    if "llvm" in target_name:
        return "llvm"
    else:
        return "cuda"


def export_library(lib: ExecutorFactoryModule, model_name: str, target_name: str,
                   work_dir: str, max_trials: int):
    """Export the libary file of a tuned network
    
    Parameters
    ----------
    lib: ExecutorFactoryModule
        The factory module created during tuning
    model_name: str 
        The name of the model
    target_name: str
        The name of the target
    work_dir: str
        The working directory the library should be saved in
    max_trials: int
        The number of trials used
    """
    simplified_target_name = get_simplified_target_name(target_name=target_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    compiled_model_name = f"{model_name}-{simplified_target_name}-{max_trials}.so"
    lib.export_library(f"{work_dir}{compiled_model_name}")
    print(f"Exported compiled library to {compiled_model_name}")


def log_config(work_dir: str):
    """Logs the configuration used for the tuning
    
    Parameters
    ----------
    work_dir: str
        The working directory to save the configuration to
    """
    message = "Configuration\n"
    message += f"Acquisition Function={configs.ACQUISITION_FUNCTION}\n"
    message += f"Kappa={configs.KAPPA}\n"
    message += f"Xi={configs.XI}\n"
    message += f"Total Trials={configs.MAX_TRIALS_LIST}\n"
    message += f"Log Limit={configs.LOG_LIMIT}\n"
    message += f"Heap={configs.USE_MIN_HEAP}\n"
    message += f"RML={configs.RESTRICTED_MEMORY_LOGGING}\n"
    message += f"SDR={configs.USE_SEQUENTIAL_DOMAIN_REDUCTION}\n"

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    file_path = os.path.join(work_dir, "configs.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(message)


def log_benchmark_results(results: BenchmarkResult, work_dir: str, num_trials: int,
                          profile_results: meta_schedule.Profiler):
    """Logs the benchmark results
    
    Parameters
    ----------
    results: BenchmarkResults
        The benchmark results
    work_dir: str
        The working directory to save the results to
    num_trials: int
        The number of tuning trials used to compile the benchmarked module
    profile_results: meta_schedule.Profiler
        The profiling results
    """
    dir_path = Path(work_dir)
    parent_dir = dir_path.parent
    if profile_results is not None:
        total_duration = float(profile_results.get()["Total"])/60
    else:
        total_duration = 0

    file_path = os.path.join(parent_dir, f"{configs.SEARCH_STRATEGY}_{configs.MODEL_NAME}.csv")

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("trials,mean,median,max,min,std,cores," \
                       "duration,kappa,log,min heap,RML,SDR,AcqFunc,Xi\n")
    with open(file_path, "+a", encoding="utf-8") as file:
        file.write(f"{num_trials},{results.mean * 1000:.4f},"
                   f"{results.median * 1000:.4f}," +
                   f"{results.max * 1000:.4f}," +
                   f"{results.min * 1000:.4f}," +
                   f"{results.std * 1000:.4f}," +
                   f"{configs.NUM_TARGET_CORES}," +
                   f"{total_duration:.4f}," +
                   f"{configs.KAPPA}," +
                   f"{configs.LOG_LIMIT}," +
                   f"{configs.USE_MIN_HEAP}," +
                   f"{configs.RESTRICTED_MEMORY_LOGGING}," +
                   f"{configs.USE_SEQUENTIAL_DOMAIN_REDUCTION}," +
                   f"{configs.ACQUISITION_FUNCTION}," +
                   f"{configs.XI}\n")



def benchmark(target: Target, lib: ExecutorFactoryModule, work_dir: str,
              graph_module: GraphModule, num_trials: int, profiler: meta_schedule.Profiler,
              number_of_benchmarks=3):
    """Benchmark the compiled model
    
    Parameters
    ----------
    target: Target
        The hardware target
    lib: ExecutorFactoryModule
        The factory module
    work_dir: str
        The working directory
    graph_module: GraphModule
        The graph module
    num_trials: int
        The number of trials used to compile the model
    profile: meta_schedule.Profiler
        The profiler
    number_of_benchmarks: int
        The number of repeated benchmarks (this is not the number of repeats per benchmark)
    """
    dev = tvm.device(str(target), 0)
    debugger = GraphModuleDebug(lib["debug_create"]("default", dev),
                                [dev], lib.get_graph_json(), work_dir)

    print("\n", debugger.profile())
    with open(f"{work_dir}debugger_profile.log", "w", encoding="utf-8") as file:
        file.write(str(debugger.profile()))

    with open(f"{work_dir}benchmark_results.log", "+a", encoding="utf-8") as file:
        for _ in range(number_of_benchmarks):
            sleep(10)
            result: BenchmarkResult = graph_module.benchmark(device=dev, repeat=10, number=1000)
            file.write(str(result))
            log_benchmark_results(result, work_dir, num_trials, profiler)
            print(result, "\n")
