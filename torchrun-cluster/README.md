# Install [metaflow](https://docs.metaflow.org/) and [metaflow-torchrun](https://github.com/outerbounds/metaflow-torchrun/tree/main)
```
pip install -r requirements.txt
cd min-gpt
```

# Set up Metaflow
- [Metaflow Resources for Engineers](https://outerbounds.com/engineering/welcome/)
- To run the Metaflow workflow as is, you need to 


## Run Karpathy's MinGPT
To run the [MinGPT](https://github.com/karpathy/minGPT) workflow using AWS Batch:
```
python flow.py --package-suffixes=.yaml run
```