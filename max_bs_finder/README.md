# Max Batch Size finder
In order to get the best performance, we need to configure the service correctly setting the `max_batch_size`
of each model in the `bentoml_runtime_configuration.yml`. Take into account that the best batch size could
change from one GPU to another and will depend on the other models that are running in that GPU at the same time.

To find the best performant batch size in an ideal case we can run the script in this folder like this:
```bash
python3 python3 run_benchmark.py --model-path ./model.onnx --output results/model.png
```

But first, install the requirements
```bash
python3 -m pip install -r requirements.txt
```