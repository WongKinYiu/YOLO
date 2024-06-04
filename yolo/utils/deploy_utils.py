import torch
from loguru import logger
from torch import Tensor

from yolo.config.config import Config
from yolo.model.yolo import create_model


class FastModelLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.compiler = self.cfg.task.fast_inference
        if self.compiler not in ["onnx", "trt"]:
            logger.warning(f"⚠️ {self.compiler} is not supported, if it is spelled wrong? Select origin model")
            self.compiler = None
        if self.cfg.device == "mps" and self.compiler == "trt":
            logger.warning("🍎 TensorRT does not support MPS devices, select origin model")
            self.compiler = None
        self.weight = cfg.weight.split(".")[0] + "." + self.compiler

    def load_model(self):
        if self.compiler == "onnx":
            logger.info("🚀 Try to use ONNX")
            return self._load_onnx_model()
        elif self.compiler == "trt":
            logger.info("🚀 Try to use TensorRT")
            return self._load_trt_model()
        else:
            return create_model(self.cfg)

    def _load_onnx_model(self):
        from onnxruntime import InferenceSession

        def onnx_forward(self: InferenceSession, x: Tensor):
            x = {self.get_inputs()[0].name: x.cpu().numpy()}
            x = [torch.from_numpy(y) for y in self.run(None, x)]
            return [x]

        InferenceSession.__call__ = onnx_forward

        try:
            ort_session = InferenceSession(self.weight, providers=["CPUExecutionProvider"])
        except Exception as e:
            logger.warning(f"🈳 Error loading ONNX model: {e}")
            ort_session = self._create_onnx_weight()
        # TODO: Update if GPU onnx unavailable change to cpu
        self.cfg.device = "cpu"
        return ort_session

    def _create_onnx_weight(self):
        from onnxruntime import InferenceSession
        from torch.onnx import export

        model = create_model(self.cfg).eval().cuda()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size)).cuda()
        export(
            model,
            dummy_input,
            self.weight,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"📥 ONNX model saved to {self.weight} ")
        return InferenceSession(self.weight, providers=["CPUExecutionProvider"])

    def _load_trt_model(self):
        from torch2trt import TRTModule

        model_trt = TRTModule()

        try:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self.weight))
        except FileNotFoundError:
            logger.warning(f"🈳 No found model weight at {self.weight}")
            model_trt = self._create_trt_weight()
        return model_trt

    def _create_trt_weight(self):
        from torch2trt import torch2trt

        model = create_model(self.cfg).eval().cuda()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size)).cuda()
        logger.info(f"♻️ Creating TensorRT model")
        model_trt = torch2trt(model, [dummy_input])
        torch.save(model_trt.state_dict(), self.weight)
        logger.info(f"📥 TensorRT model saved to {self.weight}")
        return model_trt
