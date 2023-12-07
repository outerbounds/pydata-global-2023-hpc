from metaflow import (
    FlowSpec,
    step,
    torchrun,
    current,
    batch,
    kubernetes,
    environment,
)
from decorators import gpu_profile

# import logging
# import sys
# import boto3
# import logging
# boto3.set_stream_logger(name='botocore.credentials', level=logging.ERROR)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

N_NODES = 1
N_GPU = 4
N_CPU = 8
MEMORY = 32000
SHARED_MEMORY = 4096


class CoreweaveMinGPT(FlowSpec):
    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @gpu_profile(interval=1)
    # @batch(image="eddieob/min-gpt:3", cpu=N_CPU, gpu=N_GPU, memory=MEMORY, shared_memory=SHARED_MEMORY)
    @kubernetes(image="eddieob/min-gpt:3", cpu=N_CPU, gpu=N_GPU, memory=MEMORY)
    @torchrun
    @step
    def torch_multinode(self):
        import sys
        import subprocess
        import time

        # logging.info("Installing ec2-metadata")
        # subprocess.run([sys.executable, "-m", "pip", "install", "ec2-metadata"])
        # from ec2_metadata import ec2_metadata
        # self.ec2_type = ec2_metadata.instance_type
        # logging.info(f"EC2 type: {self.ec2_type}")

        # logging.info("Running experiment with %s GPUs on %s nodes", N_GPU, N_NODES)
        t0 = time.time()
        current.torch.run(entrypoint="main.py", nproc_per_node=N_GPU)
        tf = time.time()
        self.experiment_time = tf - t0
        # logging.info("Finished experiment in %s seconds", self.experiment_time)

        self.next(self.join)

    @step
    def join(self, inputs):
        # self.times = []
        # self.ec2_types = []
        # for input in inputs:
        #     self.times.append(input.experiment_time)
        #     self.ec2_types.append(input.ec2_type)
        # logging.info("EC2 types: %s", self.ec2_types)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CoreweaveMinGPT()
