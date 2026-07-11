"""
pdf_extractor.py — PDF 提取器 + 版面分析 (v0.8)

功能：用 unstructured + Detectron2 做版面分析，输出结构化元素列表。
      支持多栏、图文混排、表格嵌入、侧边栏等复杂版面。

GPU 加速：Detectron2 模型自动使用 CUDA（如可用）。

用法：
    from src.data.extractors.pdf_extractor import PDFExtractor
    ext = PDFExtractor()
    elements = ext.extract("report.pdf")  # → List[DocumentElement]
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """版面分析后的单个文档元素。"""
    category: str       # Title / NarrativeText / Table / Image / ListItem / Header / Footer
    text: str           # 文本内容（Table 类型为 HTML 表格，Image 类型为 caption）
    page_number: int
    coordinates: Optional[tuple] = None  # (x0, y0, x1, y1)
    image_path: Optional[str] = None     # Image 类型的图片文件路径
    image_caption: Optional[str] = None  # Image 类型的 AI 生成描述
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"<{self.category} p{self.page_number}: {self.text[:60]}...>"


class PDFExtractor:
    """PDF 提取器，使用版面分析保持阅读顺序。

    两种模式：
        - fast: PyMuPDF 原生提取（快但多栏时顺序错乱）
        - hi_res: Detectron2 版面分析（慢但准确，企业级推荐）

    GPU：hi_res 模式下 Detectron2 自动使用 CUDA。
    """

    def __init__(
        self,
        mode: str = "hi_res",
        extract_images: bool = True,
        extract_tables: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            mode: "hi_res" (版面分析) 或 "fast" (PyMuPDF 原生)
            extract_images: 是否提取嵌入图片
            extract_tables: 是否检测并结构化表格
            device: "cuda" / "cpu"，None 则自动检测
        """
        self.mode = mode
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.device = device or self._detect_device()
        logger.info("PDFExtractor: mode=%s, device=%s", mode, self.device)

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def extract(self, file_path: str) -> List[DocumentElement]:
        """提取 PDF 的结构化元素。

        Args:
            file_path: PDF 文件路径

        Returns:
            DocumentElement 列表，已按阅读顺序排列
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        if self.mode == "hi_res":
            return self._extract_hi_res(file_path)
        else:
            return self._extract_fast(file_path)

    # ------------------------------------------------------------------
    # hi_res: 版面分析（推荐）
    # ------------------------------------------------------------------

    def _extract_hi_res(self, file_path: str) -> List[DocumentElement]:
        """用 unstructured + Detectron2 做版面分析。"""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            logger.warning(
                "unstructured 未安装。安装: pip install unstructured[pdf] detectron2"
            )
            return self._extract_fast(file_path)

        logger.info("版面分析中 (GPU=%s): %s", self.device, Path(file_path).name)

        # hi_res 策略 = Detectron2 检测 + 阅读顺序重排
        raw_elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=self.extract_tables,
            extract_images_in_pdf=self.extract_images,
        )

        elements = []
        for el in raw_elements:
            cat = str(el.category) if hasattr(el, "category") else "UncategorizedText"
            text = str(el) if hasattr(el, "__str__") else ""
            meta = el.metadata if hasattr(el, "metadata") else {}

            page = meta.page_number if hasattr(meta, "page_number") else 1
            coords = None
            if hasattr(meta, "coordinates"):
                pts = meta.coordinates.points if hasattr(meta.coordinates, "points") else None
                if pts:
                    coords = (pts[0][0], pts[0][1], pts[2][0], pts[2][1])

            elem = DocumentElement(
                category=cat,
                text=text.strip(),
                page_number=page or 1,
                coordinates=coords,
                metadata={"source_file": file_path, **meta.to_dict()} if hasattr(meta, "to_dict") else {},
            )
            elements.append(elem)

        # 后处理：去页眉页脚、去水印
        elements = self._postprocess(elements)

        logger.info("版面分析完成: %d 个元素 (%s)", len(elements), Path(file_path).name)
        return elements

    # ------------------------------------------------------------------
    # fast: PyMuPDF 原生（兼容旧版）
    # ------------------------------------------------------------------

    def _extract_fast(self, file_path: str) -> List[DocumentElement]:
        """用 PyMuPDF 快速提取（不做版面分析）。"""
        import fitz
        doc = fitz.open(file_path)
        elements = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                elements.append(DocumentElement(
                    category="NarrativeText",
                    text=text.strip(),
                    page_number=page_num,
                    metadata={"source_file": file_path},
                ))
        doc.close()
        return elements

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    def _postprocess(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """版面分析后处理规则。

        1. 移除页眉/页脚
        2. 移除重复水印
        3. 合并跨页段落
        """
        if not elements:
            return elements

        # ① 移除页眉/页脚
        filtered = [e for e in elements if e.category not in ("Header", "Footer")]

        # ② 检测并移除重复水印（相同文本在不同页的相同位置出现 >2 次）
        seen_texts = {}
        for e in filtered:
            key = (e.text[:100], e.coordinates)
            seen_texts[key] = seen_texts.get(key, 0) + 1

        clean = []
        for e in filtered:
            key = (e.text[:100], e.coordinates)
            if seen_texts.get(key, 0) > 2 and len(e.text) < 200:
                continue  # 高重复短文本 → 水印
            clean.append(e)

        # ③ 跨页段落合并
        merged = []
        for e in clean:
            if merged and merged[-1].category == "NarrativeText" and e.category == "NarrativeText":
                # 上一段末尾没有句号 → 可能是跨页段落
                prev_text = merged[-1].text.rstrip()
                if prev_text and prev_text[-1] not in ".!?。！？":
                    merged[-1].text = prev_text + " " + e.text
                    continue
            merged.append(e)

        return merged
