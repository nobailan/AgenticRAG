"""
provider_fallback.py — LLM 多供应商 fallback 链 (v0.7)

功能：当主 LLM 不可用时，自动切换到备用供应商，保证服务可用性。

Fallback 顺序（可配置）：
    DeepSeek → OpenAI → Anthropic

每级重试 1 次，超时 10 秒。全部失败才抛异常。

使用方式：直接替换 get_llm_response 的调用。
    from src.llm.provider_fallback import get_llm_response_with_fallback
    answer = get_llm_response_with_fallback(prompt)
"""

import logging
import os
import time
from typing import Optional, List

logger = logging.getLogger(__name__)

# 供应商配置列表：(provider_name, api_key_env, model_env)
_PROVIDER_CHAIN = [
    {
        "name": "deepseek",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "RAG_LLM_MODEL_DEEPSEEK",
        "default_model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com/v1",
    },
    {
        "name": "openai",
        "key_env": "OPENAI_API_KEY",
        "model_env": "RAG_LLM_MODEL_OPENAI",
        "default_model": "gpt-4o-mini",
        "base_url": None,  # 默认 OpenAI
    },
    {
        "name": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "RAG_LLM_MODEL_ANTHROPIC",
        "default_model": "claude-sonnet-5",
        "base_url": None,
    },
]


def get_llm_response_with_fallback(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries_per_provider: int = 1,
    timeout: int = 30,
    fallback_enabled: bool = True,
) -> str:
    """带供应商 fallback 的 LLM 调用。

    依次尝试 _PROVIDER_CHAIN 中的供应商，每个重试 max_retries_per_provider 次。
    任一成功即返回，全部失败则抛出 RuntimeError。

    Args:
        prompt: 用户 prompt
        system_prompt: 系统 prompt（可选）
        temperature: 采样温度（可选）
        max_retries_per_provider: 每个供应商的重试次数
        timeout: 每次调用的超时秒数
        fallback_enabled: 是否启用 fallback（False 时只尝试第一个供应商）

    Returns:
        LLM 响应文本

    Raises:
        RuntimeError: 所有供应商均失败
    """
    if not fallback_enabled:
        # 只尝试第一个（当前配置的）供应商
        from src.llm.llm_client import get_llm_response
        return get_llm_response(prompt, system_prompt=system_prompt, temperature=temperature)

    providers = _PROVIDER_CHAIN
    last_error = None
    tried = []

    for provider in providers:
        api_key = os.environ.get(provider["key_env"], "")
        if not api_key:
            logger.debug("跳过 %s: 未配置 API Key (%s)", provider["name"], provider["key_env"])
            continue

        model = os.environ.get(provider["model_env"], provider["default_model"])

        for attempt in range(max_retries_per_provider + 1):
            try:
                logger.info(
                    "尝试 %s/%s (attempt %d/%d)",
                    provider["name"], model, attempt + 1, max_retries_per_provider + 1,
                )

                # 直接用 openai SDK（所有供应商都兼容 OpenAI 接口）
                from openai import OpenAI

                client_kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
                base_url = provider.get("base_url")
                if base_url:
                    client_kwargs["base_url"] = base_url

                client = OpenAI(**client_kwargs)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature if temperature is not None else 0.0,
                )

                answer = response.choices[0].message.content or ""
                logger.info("%s 调用成功 (%d chars)", provider["name"], len(answer))
                return answer.strip()

            except Exception as e:
                last_error = e
                tried.append(f"{provider['name']}:{model}")
                logger.warning(
                    "%s 调用失败 (attempt %d): %s", provider["name"], attempt + 1, str(e)[:100]
                )
                if attempt < max_retries_per_provider:
                    time.sleep(1.0 * (attempt + 1))  # 指数退避

    raise RuntimeError(
        f"所有 LLM 供应商均调用失败 ({', '.join(tried) if tried else '无可用供应商'}). "
        f"最后错误: {last_error}"
    )


def get_llm_response_stream_with_fallback(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    fallback_enabled: bool = True,
):
    """带 fallback 的流式 LLM 调用。

    与 get_llm_response_with_fallback 类似，但返回 generator。
    注意：fallback 时如果切换了供应商，流式输出会短暂中断。
    """
    if not fallback_enabled:
        from src.llm.llm_client import get_llm_response_stream
        yield from get_llm_response_stream(prompt, system_prompt=system_prompt, temperature=temperature)
        return

    # 流式场景下 fallback 较复杂，先尝试主供应商
    providers = _PROVIDER_CHAIN

    for provider in providers:
        api_key = os.environ.get(provider["key_env"], "")
        if not api_key:
            continue

        model = os.environ.get(provider["model_env"], provider["default_model"])

        try:
            from openai import OpenAI
            client_kwargs = {"api_key": api_key, "timeout": 60, "max_retries": 0}
            base_url = provider.get("base_url")
            if base_url:
                client_kwargs["base_url"] = base_url

            client = OpenAI(**client_kwargs)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature if temperature is not None else 0.0,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        except Exception as e:
            logger.warning("流式调用 %s 失败: %s", provider["name"], str(e)[:100])
            continue

    # 全部失败，返回错误消息
    yield "[错误] 所有 LLM 供应商均不可用，请稍后重试。"


def get_available_providers() -> List[str]:
    """返回当前已配置 API Key 的供应商列表。"""
    available = []
    for p in _PROVIDER_CHAIN:
        if os.environ.get(p["key_env"], ""):
            available.append(p["name"])
    return available
