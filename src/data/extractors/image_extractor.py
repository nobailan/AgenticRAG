"""
image_extractor.py — 图片提取与 Captioning (v0.8)

功能：对文档中提取的图片用 BLIP-2 生成文字描述，让图片"参与检索"。

GPU 加速：BLIP-2 模型自动使用 CUDA。CPU 模式会慢 10-50 倍。

用法：
    from src.data.extractors.image_extractor import ImageCaptioner
    cap = ImageCaptioner()
    caption = cap.caption("chart.png")  # → "营收趋势图，2020-2024年持续增长"
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """BLIP-2 图片描述生成器。

    模型：Salesforce/blip2-opt-2.7b（2.7B 参数，适合单卡 GPU）
    备选：blip2-base（小模型，CPU 可用但慢）

    特性：
        - 懒加载：首次调用时才加载模型，不拖慢启动
        - GPU 自动检测：CUDA → GPU，否则 CPU（告警）
        - 批量处理：batch_caption() 一次处理多张图
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._loaded = False

        if self.device == "cpu":
            logger.warning("ImageCaptioner: 未检测到 GPU，图片描述将非常慢。")
        else:
            logger.info("ImageCaptioner: 使用 GPU (%s)", torch.cuda.get_device_name(0))

    def _load(self):
        """懒加载 BLIP-2 模型。"""
        if self._loaded:
            return
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            logger.info("正在加载 BLIP-2 模型: %s ...", self.model_name)
            self._processor = Blip2Processor.from_pretrained(self.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("BLIP-2 加载完成 (device=%s)", self.device)
        except ImportError:
            logger.error("transformers 未安装。pip install transformers")
            raise
        except Exception as e:
            logger.error("BLIP-2 加载失败: %s", e)
            raise

    def caption(self, image_path: str) -> str:
        """为单张图片生成文字描述。

        Args:
            image_path: 图片文件路径

        Returns:
            文字描述（英文），如 "a bar chart showing revenue growth from 2020 to 2024"
        """
        self._load()
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(img, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=100)
        caption = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        logger.debug("Caption: %s → %s", Path(image_path).name, caption[:80])
        return caption

    def batch_caption(self, image_paths: list) -> list:
        """批量生成图片描述（比逐个调用更快）。"""
        self._load()
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=100)
        captions = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [c.strip() for c in captions]

    def is_available(self) -> bool:
        try:
            import transformers  # noqa
            return True
        except ImportError:
            return False


# 模块级单例（模型加载慢，复用）
_captioner: Optional[ImageCaptioner] = None


def get_captioner(device: Optional[str] = None) -> ImageCaptioner:
    global _captioner
    if _captioner is None:
        _captioner = ImageCaptioner(device=device)
    return _captioner
