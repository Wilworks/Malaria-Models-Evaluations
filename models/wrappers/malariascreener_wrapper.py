"""
Wrapper Adapter for NIH MalariaScreener (BMC Infect Dis 2020).
Handles TensorFlow (.pb) and TFLite (.tflite) models for thick and thin smear classification.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
from pathlib import Path
from models.wrappers.base_wrapper import BaseModelWrapper


class MalariaScreenerWrapper(BaseModelWrapper):
    """Adapter for NIH MalariaScreener TFLite / GraphDef models."""

    def __init__(self, model_path: str, smear_type: str = "thick"):
        super().__init__(
            model_name=f"MalariaScreener_{smear_type.capitalize()}",
            model_path=model_path,
            task_type="classification"
        )
        self.smear_type = smear_type.lower()
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()

    def load_model(self) -> None:
        """Loads TFLite or TensorFlow protobuf (.pb) model."""
        path = Path(self.model_path)
        if not path.exists():
            print(f"[Warning] Model file not found at {path}")
            return

        if path.suffix == ".tflite":
            self.interpreter = tf.lite.Interpreter(model_path=str(path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[Success] Loaded TFLite model: {path.name}")
        elif path.suffix == ".pb":
            try:
                with tf.io.gfile.GFile(str(path), 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                
                self.graph = tf.Graph()
                with self.graph.as_default():
                    tf.import_graph_def(graph_def, name="")
                self.sess = tf.compat.v1.Session(graph=self.graph)
                print(f"[Success] Loaded TensorFlow GraphDef protobuf model: {path.name}")
            except Exception as e:
                print(f"[Warning] Could not load .pb model {path}: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses image to MalariaScreener exact input specs dynamically."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_w, target_h = 44, 44
        if self.input_details is not None and len(self.input_details) > 0:
            shape = self.input_details[0]['shape']
            if len(shape) == 4:
                target_h, target_w = int(shape[1]), int(shape[2])

        resized = cv2.resize(image, (target_w, target_h))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Runs zero-shot inference on input image."""
        input_data = self.preprocess_image(image)
        prob_infected = 0.0

        if self.interpreter is not None:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            prob_infected = float(output_data[0][0]) if output_data.ndim > 1 else float(output_data[0])
        elif hasattr(self, 'sess') and self.sess is not None:
            try:
                input_tensor = self.graph.get_tensor_by_name("conv2d_1_input:0")
                output_tensor = self.graph.get_tensor_by_name("dense_1/Softmax:0")
                output_data = self.sess.run(output_tensor, feed_dict={input_tensor: input_data})
                prob_infected = float(output_data[0][1]) if output_data.shape[-1] > 1 else float(output_data[0][0])
            except Exception as e:
                print(f"[Warning] Graph execution error: {e}")
                return {"predicted_class": 0, "confidence": 0.0, "error": str(e)}
        else:
            return {"predicted_class": 0, "confidence": 0.0, "error": "Model not loaded"}

        pred_class = 1 if prob_infected >= 0.5 else 0

        return {
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "confidence": prob_infected,
            "detected_objects": 1 if pred_class == 1 else 0,
            "smear_type": self.smear_type
        }
