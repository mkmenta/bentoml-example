"""Benchmark ONNX models to find best max_batch_size"""
import argparse
import timeit
import numpy as np
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
import matplotlib.pyplot as plt

import pynvml


def args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models to find best max_batch_size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--repeat", type=int, default=50,
                        help="Number of sessions in the benchmark")
    parser.add_argument("--output", type=str, default="./analysis.png",
                        help="Path to save the analysis plot")
    return parser.parse_args()


def get_gpu_vram():
    """Get GPU VRAM usage in MB."""
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    device_usage = {}
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        device_usage[i] = {
            "name": pynvml.nvmlDeviceGetName(handle),
            "free": info.free,
            "total": info.total,
            "used": info.used,
        }
    pynvml.nvmlShutdown()
    return device_usage


class Benchmark:
    def __init__(self, onnx_path: str):
        self.runner = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider']
        )
        self.batch_size = None
        self._input = None

    def set_batch_size(self, batch_size):
        """Set batch size for the benchmark."""
        self.batch_size = batch_size

    def setup(self):
        """Initialize an input tensor."""
        self._input = {}
        for input_info in self.runner.get_inputs():
            assert 'batch' in input_info.shape[0] or input_info.shape[0].startswith('unk_'), \
                f"Only batched models are supported: input_info.shape={input_info.shape}"
            self._input[input_info.name] = np.random.rand(self.batch_size, *input_info.shape[1:]).astype(np.float32)

    def run(self):
        """Run model."""
        return self.runner.run(None, self._input)


def main(args):
    """Main entry point."""
    base_usage = get_gpu_vram()

    benchmark = Benchmark(args.model_path)
    batch_sizes = [2 ** i for i in range(0, 9)]  # max = 2 ** 8 = 512
    gpu_usage = []
    ms_per_item_avg = []
    ms_per_item_std = []
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        benchmark.set_batch_size(batch_size)
        try:
            print("Running a few for initialization")
            for _ in range(5):
                benchmark.setup()
                benchmark.run()
            print(f"Running benchmark {args.repeat} iterations...")
            times = timeit.repeat(
                benchmark.run,
                # runs once at the beginning of each session, not taken into account for the timing
                setup=benchmark.setup,
                repeat=args.repeat,  # number of sessions
                number=1  # runs per session
            )
        except RuntimeException:
            print("Out of memory")
            break
        usage = get_gpu_vram()
        gpu_usage.append([(usage[i]["used"] - base_usage[i]["used"]) / 1024 // 1024
                          for i in usage])
        times = np.array(times) / batch_size
        avg = (times * 1000).mean().round().astype(int)
        std = (times * 1000).std().round().astype(int)
        ms_per_item_avg.append(avg)
        ms_per_item_std.append(std)

    # Trim batch_sizes to the number of successful runs
    batch_sizes = batch_sizes[:len(ms_per_item_avg)]

    # Print results
    print()
    for bs, avg, std, usage in zip(batch_sizes,
                                   ms_per_item_avg,
                                   ms_per_item_std,
                                   gpu_usage):
        print(f"Batch size: {bs}, ms per item: {avg} ± {std}, GPU usage: {usage}")

    # Plot
    fig = plt.figure()
    ax_top, ax_bottom = fig.subplots(nrows=2, ncols=1)

    ax_top.errorbar(batch_sizes, ms_per_item_avg, yerr=ms_per_item_std, fmt='o')
    ax_top.set_xscale('log', base=2)
    ax_top.set_xticks(batch_sizes)
    ax_top.set_title(", ".join([v['name'] for v in base_usage.values()])
                     + "\n"
                     + f"{args.model_path}",
                     fontdict={'fontsize': 8})
    ax_top.set_ylabel("ms per item")
    ax_top.grid()

    ax_bottom.plot(batch_sizes, gpu_usage)
    ax_bottom.set_xscale('log', base=2)
    ax_bottom.set_xticks(batch_sizes)
    ax_bottom.set_ylim(bottom=0)
    ax_bottom.set_ylabel("GPU memory usage (MB)")
    ax_bottom.set_xlabel("Batch size")
    ax_bottom.grid()

    plt.savefig(args.output, dpi=300)
    print(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main(args())
