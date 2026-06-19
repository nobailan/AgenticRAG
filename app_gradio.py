"""
app_gradio.py -- Gradio web UI for the Agentic RAG system (v0.3).

Provides a graphical chat interface with:
    - Streaming answer generation (token-by-token)
    - Real-time reasoning trace panel
    - Retrieved document source preview
    - Adjustable Top-K and Temperature sliders
    - Public share link (share=True)

Usage:
    python app_gradio.py

Requires:
    gradio>=4.0.0
"""
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Generator

import gradio as gr

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import env_loader
env_loader.load_env()

from src.core.config import config
from src.agents.workflow import run_workflow_streaming
from src.retrieval.retriever import is_loaded, get_chunk_count, HybridRetriever

logger = logging.getLogger(__name__)


# =============================================================================
# CSS for better UI
# =============================================================================

CUSTOM_CSS = """
.warning-banner {
    background: #332b00 !important;
    border: 1px solid #f0c040 !important;
    padding: 10px !important;
    border-radius: 8px !important;
    margin-bottom: 10px !important;
    color: #f0c040 !important;
}
.trace-box textarea {
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 13px !important;
}
"""


# =============================================================================
# Core handler — bridges Gradio UI to streaming workflow
# =============================================================================

def handle_user_message(
    message: str,
    history: List[Dict[str, str]],
    top_k: int,
    temperature: float,
) -> Generator[tuple, None, None]:
    """Process a user message through the streaming RAG pipeline.

    Called by Gradio on each user input. Yields updated (history, trace_text,
    sources_display) tuples to progressively update the UI.

    Args:
        message: The user's question.
        history: Gradio Chatbot message history.
        top_k: Top-K slider value.
        temperature: Temperature slider value.

    Yields:
        (history, trace_text, sources_display, msg_input_value) tuples.
        The 4th value clears the input box on the first yield.
    """
    # Build config overrides from UI sliders
    config_override = {
        "top_k": int(top_k),
        "llm_temperature": float(temperature),
    }

    trace_lines: List[str] = []
    sources_display = ""
    final_answer = ""
    cleared_input = False

    # Add user message to history
    history = history or []
    history.append({"role": "user", "content": message})

    # Start with an empty assistant placeholder
    history.append({"role": "assistant", "content": ""})

    try:
        for event in run_workflow_streaming(message, config_override=config_override):
            etype = event.get("type", "")

            if etype == "node":
                node = event.get("node", "")
                msg = event.get("message", "")
                detail = event.get("detail", "")
                detail_str = f" → {detail}" if detail else ""
                trace_lines.append(f"[{node}] {msg}{detail_str}")
                # Keep last 50 lines
                trace_text = "\n".join(trace_lines[-50:])

            elif etype == "token":
                token = event.get("content", "")
                final_answer = event.get("accumulated", "")
                # Stream update: replace last assistant message
                history[-1]["content"] = final_answer

            elif etype == "done":
                final_answer = event.get("final_answer", "")
                # 缓存命中标记 (v0.5)
                cache_type = event.get("cache_type", "")
                if cache_type == "exact":
                    history[-1]["content"] = final_answer + "\n\n---\n✅ 命中精确缓存"
                elif cache_type == "semantic":
                    history[-1]["content"] = final_answer + "\n\n---\n🔍 命中语义缓存"
                else:
                    history[-1]["content"] = final_answer

                # Format sources
                chunks = event.get("retrieved_chunks", [])
                if chunks:
                    sources_lines = []
                    for i, c in enumerate(chunks, 1):
                        sources_lines.append(
                            f"[{i}] {c['chunk_id']} | score={c['score']:.4f}\n"
                            f"    File: {c['source_file']}\n"
                            f"    Text: {c['text'][:150]}...\n"
                        )
                    sources_display = "\n".join(sources_lines)
                else:
                    sources_display = "No documents retrieved."

                trace_lines.append(f"[done] Answer: {len(final_answer)} chars, "
                                   f"Sources: {len(event.get('retrieved_sources', []))}")
                trace_text = "\n".join(trace_lines[-50:])
            else:
                continue

            # Yield current state — clear input box on first yield
            current_trace = "\n".join(trace_lines[-50:]) if trace_lines else ""
            msg_input_val = "" if not cleared_input else ""
            cleared_input = True
            yield history, current_trace, sources_display, msg_input_val

    except FileNotFoundError as e:
        error_msg = (
            f"❌ **索引文件未找到**\n\n"
            f"请先运行数据准备脚本生成索引：\n"
            f"```bash\n"
            f"python data_prepare.py --limit 200 --skip-ocr "
            f'--embedding-model "E:/agentProject/embedding-model/bge-base-en-v1.5"\n'
            f"```\n\n"
            f"错误详情: {e}"
        )
        history[-1]["content"] = error_msg
        trace_lines.append(f"[ERROR] Index files not found: {e}")
        yield history, "\n".join(trace_lines[-50:]), "N/A", ""

    except RuntimeError as e:
        error_msg = (
            f"❌ **API 调用失败**\n\n"
            f"请检查 API Key 是否已设置：\n"
            f"- DeepSeek: `DEEPSEEK_API_KEY`\n"
            f"- OpenAI: `OPENAI_API_KEY`\n\n"
            f"错误详情: {e}"
        )
        history[-1]["content"] = error_msg
        trace_lines.append(f"[ERROR] API Error: {e}")
        yield history, "\n".join(trace_lines[-50:]), "N/A", ""

    except Exception as e:
        error_msg = f"❌ **处理请求时发生错误**: {str(e)}"
        history[-1]["content"] = error_msg
        trace_lines.append(f"[ERROR] {traceback.format_exc()}")
        logger.error(f"handle_user_message error: {e}", exc_info=True)
        yield history, "\n".join(trace_lines[-50:]), "N/A", ""


def clear_chat() -> tuple:
    """Reset the chat interface."""
    return [], "", "", ""


# =============================================================================
# Gradio UI construction
# =============================================================================

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks UI."""

    # Check system status for banner
    index_ok = is_loaded()
    index_status = f"✅ 索引已加载 ({get_chunk_count()} chunks)" if index_ok else "⚠️ 索引未加载"

    api_key_set = bool(
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    api_status = "✅ API Key 已配置" if api_key_set else "❌ 未检测到 API Key"

    with gr.Blocks(
        title="Agentic RAG — Veracier Industries",
    ) as demo:

        # ---- Warning banner (shown if index or API missing) ----
        if not index_ok or not api_key_set:
            with gr.Row():
                warnings = []
                if not index_ok:
                    warnings.append(
                        "⚠️ 索引文件未找到，请运行: "
                        "`python data_prepare.py --limit 200 --skip-ocr "
                        '--embedding-model "E:/agentProject/embedding-model/bge-base-en-v1.5"`'
                    )
                if not api_key_set:
                    warnings.append(
                        "❌ 未检测到 API Key，请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量"
                    )
                gr.Markdown(
                    "\n\n".join(f'<div class="warning-banner">{w}</div>' for w in warnings)
                )

        # ---- Title ----
        gr.Markdown(
            "# 🏢 Agentic RAG 企业知识库 (Veracier Industries)\n"
            f"*{index_status} | {api_status}*"
        )

        # ---- Main layout: left (chat) / right (panels) ----
        with gr.Row(equal_height=False):
            # ===== LEFT COLUMN: Chat =====
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    autoscroll=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入问题，例如: What is Veracier's R&D spending in 2022?",
                        label="问题",
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清空对话", size="sm")

            # ===== RIGHT COLUMN: Info panels =====
            with gr.Column(scale=2):
                # Trace panel
                trace_box = gr.Textbox(
                    label="🧠 Agent 推理轨迹",
                    lines=18,
                    max_lines=50,
                    interactive=False,
                    placeholder="等待输入问题...",
                    elem_classes=["trace-box"],
                )

                # Sources accordion
                with gr.Accordion("📄 检索来源", open=False):
                    sources_box = gr.Textbox(
                        lines=12,
                        max_lines=30,
                        interactive=False,
                        placeholder="暂无检索结果",
                    )

                # Parameters accordion
                with gr.Accordion("⚙️ 参数调节", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=config.top_k,
                        step=1,
                        label="Top-K 检索数",
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.llm_temperature,
                        step=0.05,
                        label="Temperature",
                    )
                    gr.Markdown(
                        "*调整参数后，下次提问时生效。Top-K 越大检索越多，"
                        "Temperature 越高答案越随机。*"
                    )

        # ---- Event bindings ----
        # The handler yields 4 values: (history, trace, sources, msg_input).
        # msg_input is cleared on the very first yield so the text box
        # disappears immediately after the user clicks Send.
        send_btn.click(
            fn=handle_user_message,
            inputs=[msg_input, chatbot, top_k_slider, temp_slider],
            outputs=[chatbot, trace_box, sources_box, msg_input],
        )

        msg_input.submit(
            fn=handle_user_message,
            inputs=[msg_input, chatbot, top_k_slider, temp_slider],
            outputs=[chatbot, trace_box, sources_box, msg_input],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, trace_box, sources_box, msg_input],
        )

    return demo


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  Agentic RAG v0.3 — Gradio Web UI")
    print("=" * 60)
    print(f"  Embedding model: {config.embedding_model_name}")
    print(f"  LLM: {config.llm_provider}:{config.llm_model}")

    # Eagerly load retrieval indexes at startup (not lazy)
    print("  Loading retrieval indexes...")
    try:
        HybridRetriever()  # singleton init triggers _load_indexes()
        print(f"  Index: [OK] Loaded ({get_chunk_count()} chunks)")
    except FileNotFoundError as e:
        print(f"  Index: [WARN] {e}")
    except Exception as e:
        print(f"  Index: [ERROR] Failed to load: {e}")
    print()

    demo = build_ui()
    demo.queue(max_size=10)

    # Note: share=False because frpc cannot be downloaded behind some firewalls.
    # To enable public sharing, manually download frpc_windows_amd64_v0.3 from:
    #   https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_windows_amd64.exe
    print("  Starting Gradio server...")
    print("  Local URL: http://localhost:7860")
    print("  (If accessing from another device on LAN, use http://<your-ip>:7860)")
    print()

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue").set(  # pyright: ignore[reportPrivateImportUsage]
            body_background_fill="#0f172a",
            block_background_fill="#1e293b",
            block_border_color="#334155",
            button_primary_background_fill="#3b82f6",
        ),
        css=CUSTOM_CSS,
    )

