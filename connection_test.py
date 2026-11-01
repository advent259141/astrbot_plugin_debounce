"""Isolated, single-request diagnostics using the production classifier."""

import asyncio
import os
import re
import time

from .cloudflare_client import CloudflareError, configured_model
from .system_one import remote_classifier


MESSAGES = {
    "success": "连接成功，模型返回了有效的完整概率。",
    "unsupported": "当前使用 ONNX（本地模型），无需测试远程连接。",
    "busy": "已有连接测试正在执行，请稍候。",
    "model_busy": "模型暂时繁忙，请稍后重试。",
    "missing_key": "未配置接口密钥，请先保存配置。",
    "invalid_url": "Cloudflare 接口地址无效，请填写官方 HTTPS 账户 ai/run 基地址。",
    "invalid_model": "模型名称无效或不存在，请检查模型配置。",
    "authentication": "接口密钥无效或已失效，请检查密钥。",
    "permission": "无访问权限，请检查密钥权限与账户范围。",
    "quota": "今日免费神经元额度已用完，请检查 Cloudflare 账户额度。",
    "rate_limit": "请求受到限流，请稍后重试。",
    "service": "模型服务暂时不可用，请稍后重试。",
    "timeout": "连接测试超时，请检查网络或调整请求超时。",
    "network": "连接失败，请检查服务器到模型服务的网络。",
    "invalid_request": "模型拒绝了请求，请检查模型及接口配置。",
    "invalid_response": "模型返回格式异常，无法读取有效的完整概率。",
    "configuration": "判断后端配置无效，请检查配置并重载插件。",
}


class ConnectionTest:
    def __init__(self, config):
        self.config = config
        self.lock = asyncio.Lock()

    def info(self):
        backend = self.config.get("classifier_backend", "onnx")
        if backend == "cloudflare":
            model = configured_model(self.config) or "（无效模型）"
        elif backend == "jev":
            model = str(self.config.get("jev_model") or "jev-latest").strip()
            # Metadata is display-only; never return arbitrary configuration text.
            if not re.fullmatch(r"[A-Za-z0-9@/_.-]{1,160}", model):
                model = "（无效模型）"
            api_key = (
                str(self.config.get("jev_api_key", "") or "").strip()
                or os.getenv("TYPESAFE_API_KEY", "").strip()
            )
            if api_key and api_key in model:
                model = "（无效模型）"
        else:
            model = ""
        return {
            "backend": backend
            if backend in {"onnx", "jev", "cloudflare"}
            else "unknown",
            "model": model,
            "supported": backend in {"jev", "cloudflare"},
            "testing": self.lock.locked(),
        }

    async def run(self):
        info = self.info()
        if self.lock.locked():
            return self._result(info, "busy", 0)
        if not info["supported"]:
            code = "unsupported" if info["backend"] == "onnx" else "configuration"
            return self._result(info, code, 0)
        # No await between checking and acquiring: another request cannot queue
        # a second paid model call while the first test is running.
        async with self.lock:
            started = time.monotonic()
            try:
                classifier = remote_classifier(dict(self.config))
                probability, _ = await classifier.predict("你好，请问你能帮助我吗？")
                result = self._result(info, "success", started)
                result["probability"] = probability
                return result
            except asyncio.TimeoutError:
                code = "timeout"
            except CloudflareError as exc:
                code = exc.code
            except ValueError as exc:
                # Existing JEV transport exposes only fixed, sanitized messages.
                message = str(exc)
                if message.startswith("Configure jev_api_key"):
                    code = "missing_key"
                elif message == "JEV connection failed":
                    code = "network"
                elif message.startswith("JEV HTTP "):
                    status = message.removeprefix("JEV HTTP ")
                    code = {
                        "401": "authentication",
                        "403": "permission",
                        "404": "invalid_model",
                        "429": "rate_limit",
                    }.get(status, "service")
                else:
                    code = "invalid_response"
            except Exception:
                # Never expose exception text: it may contain a URL or credential.
                code = "invalid_response"
            return self._result(info, code, started)

    @staticmethod
    def _result(info, code, started):
        return {
            **info,
            "testing": False,
            "success": code == "success",
            "code": code,
            "message": MESSAGES.get(code, MESSAGES["invalid_response"]),
            "elapsed_ms": round((time.monotonic() - started) * 1000) if started else 0,
        }
