"""An ONNX detector."""
import bentoml
import onnx
import os

path = os.path.join(os.path.dirname(__file__), "yolov4.onnx")

saved_model = bentoml.onnx.save_model("yolo_v4",
                                      onnx.load(path),
                                      signatures={"run": {"batchable": True}})

print(f"Model saved: {saved_model}")
