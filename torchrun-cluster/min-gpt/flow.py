from metaflow import (
    FlowSpec,
    step,
    torchrun,
    current,
    batch,
    environment,
)
from decorators import gpu_profile

N_NODES = 2
N_GPU = 8
N_CPU = 48
MEMORY = 32000
SHARED_MEMORY = 4096


class MinGPT(FlowSpec):
    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @gpu_profile(interval=1)
    @batch(
        image="eddieob/min-gpt:3",
        cpu=N_CPU,
        gpu=N_GPU,
        memory=MEMORY,
        shared_memory=SHARED_MEMORY,
    )
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(entrypoint="main.py", nproc_per_node=N_GPU)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MinGPT()
