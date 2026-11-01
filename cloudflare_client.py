"""Cloudflare Workers AI's native Clef /ai/run transport."""

import asyncio
import math
import re

from .jev_client import prepare_images, validate_answers


MODELS = {
    "@cf/cloudflare/clef-flash": "clef-flash",
    "@cf/cloudflare/clef": "clef",
    "@cf/cloudflare/clef-omni": "clef-omni",
}


def configured_model(config):
    selected = str(config.get("cloudflare_model") or "@cf/cloudflare/clef-flash").strip()
    model = str(config.get("cloudflare_custom_model") or "").strip() if selected in {"custom", "自定义"} else selected
    if selected not in MODELS and selected not in {"custom", "自定义"}:
        return None
    if not re.fullmatch(r"@(cf|hf)/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", model):
        return None
    key = str(config.get("cloudflare_api_key") or "").strip()
    return None if key and key in model else model


class CloudflareError(ValueError):
    """A public error category; never retain response bodies or credentials."""

    def __init__(self, code):
        self.code = code
        super().__init__(f"Cloudflare {code}")


class CloudflareClient:
    def __init__(self, config):
        self.api_key = str(config.get("cloudflare_api_key", "") or "").strip()
        self.model = configured_model(config)
        base_url = str(config.get("cloudflare_base_url", "") or "").strip().rstrip("/")
        # Reject userinfo, alternate hosts, ports, queries and redirects so that
        # a saved typo cannot send the credential to an unrelated service.
        self.valid_base_url = bool(
            re.fullmatch(
                r"https://api\.cloudflare\.com/client/v4/accounts/[a-fA-F0-9]{32}/ai/run",
                base_url,
            )
        )
        self.endpoint = base_url + "/" + (self.model or "")
        self.enable_images = bool(config.get("cloudflare_enable_images", False))
        try:
            self.max_images = max(
                1, min(4, int(config.get("cloudflare_max_images", 4)))
            )
        except (TypeError, ValueError, OverflowError):
            self.max_images = 4
        try:
            timeout = float(config.get("cloudflare_timeout_seconds", 10))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            timeout = 10.0
        self.timeout = min(timeout, 120.0)

    async def evaluate(self, state, questions: dict, *, images=None) -> dict:
        if not self.api_key:
            raise CloudflareError("missing_key")
        if not self.valid_base_url:
            raise CloudflareError("invalid_url")
        if self.model is None:
            raise CloudflareError("invalid_model")
        return await asyncio.wait_for(
            self._evaluate(state, questions, images), timeout=self.timeout
        )

    async def _evaluate(self, state, questions, images):
        import aiohttp

        payload = {"model": self.model.rsplit("/", 1)[-1], "state": state, "questions": questions}
        if images and self.enable_images:
            prepared, omitted = await prepare_images(images, self.max_images)
            sizes = [
                len(image["data_base64"]) * 3 // 4 - image["data_base64"].count("=")
                for image in prepared
            ]
            if (
                any(size > 4 * 1024 * 1024 for size in sizes)
                or sum(sizes) > 8 * 1024 * 1024
            ):
                raise CloudflareError("invalid_request")
            # Clef accepts embedded image data URLs, not TypeSafe's assets map.
            payload["images"] = [
                "data:image/jpeg;base64," + image["data_base64"] for image in prepared
            ]
            if omitted:
                payload["state"] = (
                    dict(state) if isinstance(state, dict) else {"message": state}
                )
                payload["state"]["omitted_images"] = omitted
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    allow_redirects=False,
                ) as response:
                    try:
                        data = await response.json()
                    except (ValueError, aiohttp.ContentTypeError):
                        data = None
                    if response.status != 200:
                        raise CloudflareError(_error_code(response.status, data))
        except aiohttp.ClientError:
            raise CloudflareError("network") from None
        if not isinstance(data, dict) or data.get("success") is not True:
            raise CloudflareError(_error_code(200, data))
        result = data.get("result")
        try:
            return validate_answers(result, questions)
        except ValueError:
            raise CloudflareError("invalid_response") from None


def _error_code(status, data):
    errors = data.get("errors") if isinstance(data, dict) else None
    codes = (
        {
            error.get("code")
            for error in errors
            if isinstance(error, dict) and type(error.get("code")) is int
        }
        if isinstance(errors, list)
        else set()
    )
    if status == 401:
        return "authentication"
    if status == 403:
        return "permission"
    if codes & {3036}:
        return "quota"
    if codes & {3040}:
        return "model_busy"
    if status == 404 or codes & {3042, 5007}:
        return "invalid_model"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "service"
    if status == 400:
        return "invalid_request"
    return "invalid_response"
