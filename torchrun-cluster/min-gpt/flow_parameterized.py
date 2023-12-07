from metaflow import (
    FlowSpec,
    step,
    torchrun,
    current,
    batch,
    kubernetes,
    environment,
    Parameter,
)
from decorators import gpu_profile

N_NODES = 4
N_GPU = 4
N_CPU = 48
MEMORY = 32000
SHARED_MEMORY = 4096


@trigger
class MinGPT(FlowSpec):
    n_layer = Parameter("n_layer", help="Number of layers", default=8)
    n_head = Parameter("n_head", help="Number of heads", default=8)
    n_embd = Parameter("n_embd", help="Embedding size", default=512)
    max_epochs = Parameter("max_epochs", help="Max epochs", default=10)
    batch_size = Parameter("batch_size", help="Batch size", default=256)
    data_loader_workers = Parameter(
        "data_loader_workers", help="Data loader workers", default=8
    )
    weight_decay = Parameter("weight_decay", help="Weight decay", default=0.1)
    learning_rate = Parameter("learning_rate", help="Learning rate", default=0.0003)

    def _set_config_overrides():
        with open("gpt2_train_cfg.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["gpt_config"]["n_layer"] = self.n_layer
        config_dict["gpt_config"]["n_head"] = self.n_head
        config_dict["gpt_config"]["n_embd"] = self.n_embd
        config_dict["trainer_config"]["max_epochs"] = self.max_epochs
        config_dict["trainer_config"]["batch_size"] = self.batch_size
        config_dict["trainer_config"]["data_loader_workers"] = self.data_loader_workers
        config_dict["optimizer_config"]["weight_decay"] = self.weight_decay
        config_dict["optimizer_config"]["learning_rate"] = self.learning_rate
        with open("gpt2_train_cfg.yaml", "w") as f:
            yaml.dump(config_dict, f)

    @step
    def start(self):
        self._set_config_overrides()
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
