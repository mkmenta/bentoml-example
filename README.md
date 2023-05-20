
# BentoML example
Serving of YOLOv4 from https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb using [BentoML](https://github.com/bentoml/BentoML).
## Benchmark results
All numbers are in **milliseconds** except for the Total time which is in seconds.

These numbers were obtained in an AWS EC2 g4dn.xlarge with a Tesla T4.
### Case 1: max_latency_ms = 500 milliseconds

|                        |       |      |       |      |       |
|------------------------|-------|------|-------|------|-------|
|Total time (s)          |92     |      |       |      |       |
|Requests per second     |49     |      |       |      |       |
|Milliseconds per request|20     |      |       |      |       |

|                        |       |      |       |      |       |
|------------------------|-------|------|-------|------|-------|
|Response codes:         |       |      |       |      |       |
|200                     |2130   |47.33%|       |      |       |
|503                     |2370   |52.67%|       |      |       |

|Key                     |Mean   |Std   |Median |Min   |Max    |
|------------------------|-------|------|-------|------|-------|
|elapsed_200             |2054.95|564.71|1969.34|783.54|4531.79|
|elapsed_error           |1672.89|568.3 |1578.83|403.17|3726.89|
|fetch_image             |171.1  |163.19|114.93 |16.06 |1070.25|
|preprocess              |5.67   |4.91  |3.42   |2.09  |59.36  |
|compute                 |1167.88|427.9 |1161.81|177.79|3233.98|
|postprocess             |18.8   |8.08  |16.2   |6.88  |63.64  |


### Case 2: max_latency_ms = infinite (1000 seconds)

|                        |       |      |       |      |       |
|------------------------|-------|------|-------|------|-------|
|Total time (s)          |145    |      |       |      |       |
|Requests per second     |31     |      |       |      |       |
|Milliseconds per request|32     |      |       |      |       |

|                        |       |      |       |      |       |
|------------------------|-------|------|-------|------|-------|
|Response codes:         |       |      |       |      |       |
|200                     |4500   |100.00%|       |      |       |

|Key                     |Mean   |Std   |Median |Min   |Max    |
|------------------------|-------|------|-------|------|-------|
|elapsed_200             |3162.73|11978.43|1212.88|367.26|145022.34|
|fetch_image             |40.14  |69.11 |27.49  |12.55 |942.42 |
|preprocess              |3.94   |2.46  |2.99   |1.82  |30.1   |
|compute                 |3022.88|11947.19|1081.83|312.66|144292.8|
|postprocess             |16.1   |6.47  |14.13  |6.85  |78.98  |

## How to run
### Setup
1. Download `https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx`
2. `python3 -m pip install -r requirements.txt `
3. `python3 make_model.py`
4. `python3 -m pip install -r requirements-service.txt `
5. `bentoml build`
6. `bentoml containerize detect:latest`

### Run
First start the service. Replace `[CONF_FILE]` by one of the following files to reproduce each of the cases:

- `case1_runtime_configuration.yaml`: 500ms of max_latency_ms
- `case2_runtime_configuration.yaml`: "infinite" max_latency_ms

```
docker run --gpus all -it --rm -p 3000:3000 \
-v $(pwd)/[CONF_FILE]:/home/bentoml/configuration.yml \
-e BENTOML_CONFIG=/home/bentoml/configuration.yml \
detect:XXXXXX serve --production
```

Then run the benchmark:
```
python3 benchmark.py
```