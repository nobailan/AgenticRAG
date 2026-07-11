"""
pipeline.py — v0.8 数据准备主流水线

串联：提取 → 清洗 → Captioning → 分块 → Embedding → 索引

所有可用环节均使用 GPU 加速（版面分析 / BLIP-2 / Embedding）。

用法：
    python -m src.data.pipeline --input ./my_docs --limit 50
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import config

logger = logging.getLogger(__name__)


class DataPipeline:
    """v0.8 数据准备流水线。

    流程：
        1. 文件扫描（PDF/Word/Excel/PPT/HTML/图片）
        2. 按格式分发到对应提取器
        3. 图片 Captioning（BLIP-2, GPU）
        4. 四层清洗
        5. 自适应分块（Sentence Window）
        6. Embedding 生成（GPU）
        7. FAISS + BM25 索引构建
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if self._has_gpu() else "cpu")
        self.stats = {
            "files_scanned": 0, "files_processed": 0, "files_failed": 0,
            "elements_extracted": 0, "images_captioned": 0,
            "chunks_generated": 0, "total_time": 0,
        }

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Step 1: 文件扫描与分类
    # ------------------------------------------------------------------

    _FORMAT_MAP = {
        ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
        ".xlsx": "xlsx", ".xls": "xlsx",
        ".pptx": "pptx", ".ppt": "pptx",
        ".html": "html", ".htm": "html",
        ".txt": "text", ".md": "text", ".csv": "text",
        ".png": "image", ".jpg": "image", ".jpeg": "image",
        ".tiff": "image", ".bmp": "image",
    }

    def scan_files(self, input_dir: str, limit: int = 0) -> List[Dict[str, str]]:
        """扫描目录，返回 (path, format, doc_type) 列表。"""
        files = []
        root = Path(input_dir)
        if not root.exists():
            logger.error("目录不存在: %s", input_dir)
            return []

        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in self._FORMAT_MAP:
                fmt = self._FORMAT_MAP[f.suffix.lower()]
                doc_type = self._guess_doc_type(f)
                files.append({"path": str(f), "format": fmt, "doc_type": doc_type})

            if limit and len(files) >= limit:
                break

        self.stats["files_scanned"] = len(files)
        logger.info("扫描完成: %d 文件", len(files))
        return files

    def _guess_doc_type(self, filepath: Path) -> str:
        """从路径/文件名猜测文档类型。"""
        name = filepath.name.lower()
        parent = filepath.parent.name.lower()
        full = str(filepath).lower()

        if any(kw in full for kw in ["contract", "contrat", "msa", "agreement", "accord"]):
            return "contract"
        if any(kw in full for kw in ["finance", "financial", "fiscal", "ifrs", "report",
                                      "bilan", "compte", "annual"]):
            return "financial"
        if any(kw in full for kw in ["patent", "brevet", "technical", "technique",
                                      "spec", "manual", "sop"]):
            return "technical"
        if any(kw in full for kw in ["email", "courrier", "correspond", "letter"]):
            return "email"
        if any(kw in full for kw in ["policy", "politique", "charter", "accord",
                                      "procedure", "regulation"]):
            return "policy"
        return "generic"

    # ------------------------------------------------------------------
    # Step 2-4: 提取 + 清洗 + Captioning
    # ------------------------------------------------------------------

    def process_file(self, file_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """处理单个文件：提取 → 清洗 → 分块。

        Args:
            file_info: {"path", "format", "doc_type"}

        Returns:
            该文件的 chunk 列表
        """
        path = file_info["path"]
        fmt = file_info["format"]
        doc_type = file_info["doc_type"]
        logger.debug("处理: %s (%s, %s)", Path(path).name, fmt, doc_type)

        try:
            # ---- 提取 ----
            elements = self._extract(path, fmt)
            self.stats["elements_extracted"] += len(elements)

            # ---- Captioning (图片) ----
            elements = self._caption_images(elements)

            # ---- 清洗 ----
            from src.data.cleaners.document_cleaner import DocumentCleaner
            cleaner = DocumentCleaner()
            elements = cleaner.clean(elements)

            # ---- 分块 ----
            from src.data.chunkers.adaptive_chunker import AdaptiveChunker
            chunker = AdaptiveChunker(doc_type=doc_type, mode="sentence_window")
            chunks = chunker.chunk(elements, metadata={
                "source_file": path,
                "doc_type": doc_type,
                "format": fmt,
            })
            self.stats["chunks_generated"] += len(chunks)
            self.stats["files_processed"] += 1

            return chunks

        except Exception as e:
            logger.error("处理失败 %s: %s", Path(path).name, e)
            self.stats["files_failed"] += 1
            return []

    # ------------------------------------------------------------------
    # 提取器分发
    # ------------------------------------------------------------------

    def _extract(self, path: str, fmt: str):
        if fmt == "pdf":
            from src.data.extractors.pdf_extractor import PDFExtractor
            ext = PDFExtractor(mode="hi_res", device=self.device)
            return ext.extract(path)
        elif fmt in ("docx", "xlsx", "pptx"):
            from src.data.extractors.office_extractor import OfficeExtractor
            return OfficeExtractor().extract(path)
        elif fmt == "image":
            from PIL import Image
            from src.data.extractors.pdf_extractor import DocumentElement
            img = Image.open(path)
            return [DocumentElement(
                category="Image", text="", page_number=1,
                image_path=path,
                metadata={"source_file": path, "format": "image",
                          "width": img.width, "height": img.height},
            )]
        else:
            # HTML / text
            from src.data.extractors.pdf_extractor import DocumentElement
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            return [DocumentElement(
                category="NarrativeText", text=text[:50000], page_number=1,
                metadata={"source_file": path, "format": fmt},
            )]

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def _caption_images(self, elements):
        """为 Image 元素生成 BLIP-2 描述。"""
        images = [e for e in elements if e.category == "Image" and e.image_path]
        if not images:
            return elements

        try:
            from src.data.extractors.image_extractor import get_captioner
            captioner = get_captioner(device=self.device)
            if not captioner.is_available():
                logger.warning("BLIP-2 不可用，跳过图片 Captioning")
                return elements

            paths = [e.image_path for e in images]
            captions = captioner.batch_caption(paths)

            for el, cap in zip(images, captions):
                el.image_caption = cap
                el.text = f"[IMAGE: {cap}]"
                self.stats["images_captioned"] += 1

            logger.info("Captioning 完成: %d 张图片", len(images))
        except Exception as e:
            logger.warning("Captioning 失败: %s", e)

        return elements

    # ------------------------------------------------------------------
    # Step 5-6: Embedding + 索引
    # ------------------------------------------------------------------

    def build_indexes(self, all_chunks: List[Dict], output_dir: Optional[str] = None):
        """对所有 chunk 生成 embedding 并构建 FAISS + BM25 索引。"""
        if not all_chunks:
            logger.error("无 chunk，跳过索引构建")
            return

        output_dir = Path(output_dir) if output_dir else Path("data/indices")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Embedding（GPU 加速）
        logger.info("Embedding %d chunks (device=%s)...", len(all_chunks), self.device)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.embedding_model_name, device=self.device)
        texts = [c.get("expanded_text", c["text"]) for c in all_chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS 索引
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(output_dir / "faiss.index"))
        logger.info("FAISS 索引: %d vectors, dim=%d", len(all_chunks), dim)

        # BM25 索引 + chunks.jsonl
        import json
        from src.data.data_prepare import build_bm25_index

        chunks_path = output_dir / "chunks.jsonl"
        legacy_chunks = []
        for i, c in enumerate(all_chunks):
            legacy_chunks.append({
                "chunk_id": f"doc_{i}",
                "text": c.get("expanded_text", c["text"]),
                "metadata": c.get("metadata", {}),
                "embedding_idx": i,
            })
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in legacy_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("Chunks JSONL: %d chunks -> %s", len(legacy_chunks), chunks_path)

        build_bm25_index(legacy_chunks, output_dir)

        logger.info("索引构建完成: %s", output_dir)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, input_dir: str, output_dir: Optional[str] = None, limit: int = 0):
        """运行完整流水线。

        Args:
            input_dir: 输入文档目录
            output_dir: 输出目录
            limit: 限制处理文件数（0 = 全部）
        """
        t0 = time.time()
        logger.info("=" * 50)
        logger.info("AgenticRAG v0.8 数据流水线")
        logger.info(f"输入: {input_dir}  设备: {self.device}")
        logger.info("=" * 50)

        # 1. 扫描
        files = self.scan_files(input_dir, limit=limit)
        if not files:
            return

        # 2-4. 逐文件处理
        all_chunks = []
        for i, f in enumerate(files, 1):
            print(f"\r[{i}/{len(files)}] {Path(f['path']).name[:60]}...", end="", flush=True)
            chunks = self.process_file(f)
            all_chunks.extend(chunks)
        print()

        # 5-6. 索引
        self.build_indexes(all_chunks, output_dir)

        self.stats["total_time"] = time.time() - t0
        self._print_summary()

    def _print_summary(self):
        s = self.stats
        print("\n" + "=" * 50)
        print("流水线完成")
        print(f"  文件: {s['files_scanned']} 扫描, "
              f"{s['files_processed']} 成功, {s['files_failed']} 失败")
        print(f"  元素: {s['elements_extracted']} 提取")
        print(f"  图片: {s['images_captioned']} Captioned")
        print(f"  Chunks: {s['chunks_generated']} 生成")
        print(f"  耗时: {s['total_time']:.0f}s ({s['total_time']/60:.1f}min)")
        print(f"  设备: {self.device}")
        print("=" * 50)


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgenticRAG v0.8 数据流水线")
    parser.add_argument("--input", "-i", required=True, help="输入文档目录")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="限制文件数")
    parser.add_argument("--device", default=None, help="cuda/cpu（默认自动）")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipe = DataPipeline(device=args.device)
    pipe.run(input_dir=args.input, output_dir=args.output, limit=args.limit)
