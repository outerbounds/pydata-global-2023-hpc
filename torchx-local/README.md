# Goal
- Intro to running distributed PyTorch code using [torchx](https://pytorch.org/torchx/latest/quickstart.html)
- Quickstart for running torchx jobs on Kubernetes using [Volcano](https://github.com/volcano-sh/volcano)

# Setup instructions 
```
cd torchx-local
```

### [Install Minikube](https://minikube.sigs.k8s.io/docs/start/)

For example, on Mac:
```
brew install minikube
```

### Run minikube
```
minikube start
```

### List available nodes in the cluster
```
kubectl get nodes
```

### Find available resources (e.g., compute and memory) in the cluster
```
kubectl describe node <node_name>
```

### Run Minikube dashboard
```
minikube dashboard
```

### [Install Volcano](https://github.com/volcano-sh/volcano)
```
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
```

### Install torchx
```
pip install -r requirements.txt
```

# Local Docker demo
```
torchx run --scheduler local_docker utils.python --script hello_world.py "PyData Global"
```
 
# Local Kubernetes demo (with Dockerhub)

### Create repository in Dockerhub
This scheduler will package up the local workspace as a layer on top of the specified image. This provides a very similar environment to the container based remote schedulers.

Go to the `.torchxconfig` file in this repository and change the `image` line to your repository and tag.

### Create a queue in Volcano
```
kubectl apply -f queue.yaml
```

### Run a torch distributed job on Volcano
```
torchx run --scheduler kubernetes utils.python --script hello_world.py "PyData"
```
The above command will run a Kubernetes Pod! Now, let's check the logs:
```
kubectl logs <POD ID>
```