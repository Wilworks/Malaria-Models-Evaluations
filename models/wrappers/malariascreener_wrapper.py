"""
Wrapper Adapter for NIH MalariaScreener Models (BMC Infectious Diseases, 2020).
Handles MobileNetV2 TensorFlow GraphDef (.pb) and TensorFlow Lite (.tflite) models
for thick and thin smear zero-shot classification.
Supports Google's lightweight ai-edge-litert / tflite_runtime as well as full TensorFlow.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

# Prefer lightweight TFLite runtime, fallback to full tensorflow
try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        try:
            import tensorflow as tf
            TFLiteInterpreter = tf.lite.Interpreter
        except ImportError:
            TFLiteInterpreter = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from models.wrappers.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class MalariaScreenerWrapper(BaseModelWrapper):
    """Adapter for NIH MalariaScreener TFLite and GraphDef models."""

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
        self.graph = None
        self.sess = None
        self.load_model()

    def load_model(self) -> None:
        """Loads TFLite or TensorFlow protobuf (.pb) model."""
        path = Path(self.model_path)
        if not path.exists():
            logger.warning("Model file not found at: %s", path)
            return

        if path.suffix == ".tflite":
            if TFLiteInterpreter is None:
                logger.error("No TFLite runtime available. Please install ai-edge-litert or tensorflow.")
                return
            try:
                self.interpreter = TFLiteInterpreter(model_path=str(path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info("Loaded TFLite model: %s", path.name)
            except Exception as e:
                logger.error("Failed to initialize TFLite model %s: %s", path.name, e)

        elif path.suffix == ".pb":
            if tf is None:
                logger.warning("Full TensorFlow is required to run .pb GraphDef model %s", path.name)
                return
            try:
                with tf.io.gfile.GFile(str(path), "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())

                self.graph = tf.Graph()
                with self.graph.as_default():
                    tf.import_graph_def(graph_def, name="")
                self.sess = tf.compat.v1.Session(graph=self.graph)
                logger.info("Loaded TensorFlow GraphDef protobuf model: %s", path.name)
            except Exception as e:
                logger.warning("Could not load .pb model %s: %s", path.name, e)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses image to MalariaScreener expected input dimensions and normalization."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_w, target_h = 44, 44
        if self.input_details is not None and len(self.input_details) > 0:
            shape = self.input_details[0]["shape"]
            if len(shape) == 4:
                target_h, target_w = int(shape[1]), int(shape[2])

        resized = cv2.resize(image, (target_w, target_h))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Runs zero-shot inference on input micrograph array."""
        input_data = self.preprocess_image(image)
        prob_infected = 0.0

        if self.interpreter is not None:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
            # Output shape [1, 2]: index 0 = uninfected, index 1 = infected
            if output_data.shape[-1] > 1:
                prob_infected = float(output_data[0][1])
            else:
                prob_infected = float(output_data[0][0])
        elif self.sess is not None:
            try:
                input_tensor = self.graph.get_tensor_by_name("conv2d_1_input:0")
                output_tensor = self.graph.get_tensor_by_name("dense_1/Softmax:0")
                output_data = self.sess.run(output_tensor, feed_dict={input_tensor: input_data})
                # Output shape [1, 2]: index 0 = uninfected, index 1 = infected
                prob_infected = float(output_data[0][1]) if output_data.shape[-1] > 1 else float(output_data[0][0])
            except Exception as e:
                logger.warning("Graph execution error for %s: %s", self.model_name, e)
                return {
                    "model_name": self.model_name,
                    "predicted_class": 0,
                    "confidence": 0.0,
                    "error": str(e)
                }
        else:
            return {
                "model_name": self.model_name,
                "predicted_class": 0,
                "confidence": 0.0,
                "error": "Model not loaded"
            }

        pred_class = 1 if prob_infected >= 0.5 else 0

        return {
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "confidence": round(prob_infected if pred_class == 1 else (1.0 - prob_infected), 4),
            "probability_infected": round(prob_infected, 4),
            "smear_type": self.smear_type
        }
