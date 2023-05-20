from typing import Dict
import bentoml
from bentoml.io import JSON
import numpy as np

from utils import (StopWatch,
                   image_preprocess,
                   nms,
                   postprocess_bbbox,
                   postprocess_boxes,
                   read_image_from_url)

runner = bentoml.onnx.get("yolo_v4:latest").to_runner()

svc = bentoml.Service("detect", runners=[runner])
ANCHORS = np.array([[[12.0, 16.0], [19.0, 36.0], [40.0, 28.0]], [[36.0, 75.0], [76.0, 55.0],
                   [72.0, 146.0]], [[142.0, 110.0], [192.0, 243.0], [459.0, 401.0]]])
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

STRIDES = np.array(STRIDES)
INPUT_SIZE = 416


@svc.api(input=JSON(), output=JSON())
async def detect(data: Dict) -> Dict:
    stopwatch = StopWatch()
    stopwatch.start_stop("fetch_image")
    original_image = await read_image_from_url(data["image_url"])
    stopwatch.start_stop("fetch_image")

    stopwatch.start_stop("preprocess")
    original_image_size = original_image.shape[:2]

    image_data = image_preprocess(np.copy(original_image), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    stopwatch.start_stop("preprocess")

    stopwatch.start_stop("compute")
    detections = await runner.async_run(image_data)
    stopwatch.start_stop("compute")

    stopwatch.start_stop("postprocess")
    detections = [d.copy() for d in detections]
    pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size, INPUT_SIZE, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')
    stopwatch.start_stop("postprocess")
    return {"bboxes": [bbox.tolist() for bbox in bboxes],
            "timings": stopwatch.get_timing()}
