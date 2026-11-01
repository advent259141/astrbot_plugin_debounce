import asyncio
import importlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web

from test_main import debounce
import test_jev_client as jevtests


cf = importlib.import_module("debounce_test_module.cloudflare_client")


class CloudflareHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.status = 200
        self.data = {
            "success": True,
            "result": {"answers": {"complete": {"type": "noul", "noul": 0.9}}},
        }
        self.invalid_json = False
        self.delay = 0

        async def handler(request):
            self.requests.append((request.path, request.headers, await request.json()))
            await asyncio.sleep(self.delay)
            if self.invalid_json:
                return web.Response(
                    text="secret-response",
                    content_type="application/json",
                    status=self.status,
                )
            return web.json_response(self.data, status=self.status)

        app = web.Application()
        app.router.add_post("/{tail:.*}", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.local = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.config = {
            "cloudflare_base_url": "https://api.cloudflare.com/client/v4/accounts/"
            + "0" * 32
            + "/ai/run/",
            "cloudflare_api_key": "test-only-key",
            "cloudflare_enable_images": True,
        }
        self.client = cf.CloudflareClient(self.config)
        self.client.endpoint = self.local + "/run"
        self.questions = {"complete": {"type": "noul"}}

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_all_models_use_full_url_and_short_body_selector(self):
        for model, selector in cf.MODELS.items():
            client = cf.CloudflareClient({**self.config, "cloudflare_model": model})
            self.assertTrue(client.endpoint.endswith("/ai/run/" + model))
            client.endpoint = self.local + "/" + model
            answers = await client.evaluate({"message": "你好"}, self.questions)
            path, headers, body = self.requests[-1]
            self.assertEqual(path, "/" + model)
            self.assertEqual(headers["Authorization"], "Bearer test-only-key")
            self.assertEqual(
                body,
                {
                    "model": selector,
                    "state": {"message": "你好"},
                    "questions": self.questions,
                },
            )
            self.assertEqual(answers["complete"]["noul"], 0.9)

    async def test_custom_model_uses_full_id_and_short_selector(self):
        config = {**self.config, "cloudflare_model": "自定义", "cloudflare_custom_model": "@cf/cloudflare/clef-next"}
        client = cf.CloudflareClient(config)
        self.assertTrue(client.endpoint.endswith("/@cf/cloudflare/clef-next"))
        client.endpoint = self.local + "/custom"
        await client.evaluate("x", self.questions)
        self.assertEqual(self.requests[-1][2]["model"], "clef-next")
        for model in ["", "https://evil.example", "@cf/cloudflare/../x", "@cf/cloudflare/x?key=y", "@cf/cloudflare/x#y", "@cf/cloudflare/test-only-key"]:
            with self.assertRaisesRegex(cf.CloudflareError, "invalid_model"):
                await cf.CloudflareClient({**config, "cloudflare_custom_model": model}).evaluate("x", self.questions)
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(cf.CloudflareClient({}).enable_images)

    async def test_invalid_configuration_never_sends_request(self):
        bad_urls = [
            "http://api.cloudflare.com/client/v4/accounts/" + "0" * 32 + "/ai/run",
            "https://api.cloudflare.com.evil.example/ai/run",
            self.config["cloudflare_base_url"] + "?token=test-only-key",
            self.config["cloudflare_base_url"].replace(
                "api.cloudflare.com", "user@api.cloudflare.com"
            ),
            self.config["cloudflare_base_url"].replace(
                "api.cloudflare.com", "api.cloudflare.com:443"
            ),
            self.config["cloudflare_base_url"].replace("0" * 32, "bad-account"),
        ]
        for base in bad_urls:
            client = cf.CloudflareClient({**self.config, "cloudflare_base_url": base})
            with self.assertRaisesRegex(cf.CloudflareError, "invalid_url"):
                await client.evaluate("x", self.questions)
        for config, code in [
            ({"cloudflare_api_key": ""}, "missing_key"),
            ({"cloudflare_model": "@cf/cloudflare/nonexistent"}, "invalid_model"),
        ]:
            with self.assertRaisesRegex(cf.CloudflareError, code):
                await cf.CloudflareClient({**self.config, **config}).evaluate(
                    "x", self.questions
                )
        self.assertEqual(self.requests, [])

    def test_timeout_and_image_limit_configuration(self):
        for value in [None, "invalid", float("nan"), float("inf"), -1, 0]:
            client = cf.CloudflareClient(
                {**self.config, "cloudflare_timeout_seconds": value}
            )
            self.assertEqual(client.timeout, 10)
        self.assertEqual(
            cf.CloudflareClient({"cloudflare_timeout_seconds": 999}).timeout, 120
        )
        for value, expected in [(0, 1), (99, 4), ("bad", 4), (2, 2)]:
            client = cf.CloudflareClient({"cloudflare_max_images": value})
            self.assertEqual(client.max_images, expected)

    async def test_oversized_prepared_images_do_not_send_partial_requests(self):
        for prepared in [
            [{"data_base64": "A" * (6 * 1024 * 1024)}],
            [{"data_base64": "A" * (4 * 1024 * 1024)}] * 3,
        ]:
            with patch.object(
                cf, "prepare_images", AsyncMock(return_value=(prepared, 0))
            ):
                with self.assertRaisesRegex(cf.CloudflareError, "invalid_request"):
                    await self.client.evaluate(
                        "image", self.questions, images=[object()]
                    )
        self.assertEqual(self.requests, [])

    async def test_http_errors_use_safe_categories(self):
        for status, errors, expected in [
            (401, [], "authentication"),
            (403, [], "permission"),
            (404, [], "invalid_model"),
            (429, [{"code": 3036}], "quota"),
            (429, [{"code": 3040}], "model_busy"),
            (429, [], "rate_limit"),
            (500, [], "service"),
            (400, [{"code": 5007}], "invalid_model"),
            (400, [], "invalid_request"),
            (302, [], "invalid_response"),
            (200, [{"code": 3036}], "quota"),
        ]:
            self.status = status
            self.data = {"success": False, "errors": errors, "message": "test-only-key"}
            with self.assertRaises(cf.CloudflareError) as caught:
                await self.client.evaluate("x", self.questions)
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn("test-only-key", str(caught.exception))
        self.assertEqual(len(self.requests), 11)  # No retries or redirects.

    async def test_invalid_success_envelopes_and_answer_shapes_fail(self):
        for data in [
            [],
            {},
            {"success": 1},
            {"success": True, "result": []},
            {"success": True, "result": {"answers": {}}},
            {"success": True, "result": {"answers": {"complete": {"type": "score"}}}},
        ]:
            self.data = data
            with self.assertRaisesRegex(cf.CloudflareError, "invalid_response"):
                await self.client.evaluate("x", self.questions)
        self.invalid_json = True
        with self.assertRaisesRegex(cf.CloudflareError, "invalid_response"):
            await self.client.evaluate("x", self.questions)

    async def test_probability_is_strict_and_never_requires_one(self):
        classifier = debounce.SystemOneSentenceClassifier(self.client)
        for value in [True, None, "0.9", -0.1, 1.1, float("nan"), float("inf")]:
            self.data["result"]["answers"]["complete"]["noul"] = value
            with self.assertRaises(ValueError):
                await classifier.predict("你好")
        self.data["result"]["answers"]["complete"]["noul"] = 0.42
        self.assertEqual(await classifier.predict("你好"), (0.42, 0.42))

    async def test_timeout_and_network_failures(self):
        self.client.timeout = 0.01
        self.delay = 0.05
        with self.assertRaises(asyncio.TimeoutError):
            await self.client.evaluate("x", self.questions)
        with patch.object(
            aiohttp,
            "ClientSession",
            side_effect=aiohttp.ClientConnectionError("test-only-key"),
        ):
            with self.assertRaisesRegex(cf.CloudflareError, "network") as caught:
                await self.client.evaluate("x", self.questions)
        self.assertNotIn("test-only-key", str(caught.exception))

    async def test_images_are_embedded_limited_and_preserve_original_state(self):
        image_factory = jevtests.JevHTTPTests.image
        pictures = [image_factory(self, source=str(index)) for index in range(6)]
        state = {"message": "图片"}
        await self.client.evaluate(
            state, self.questions, images=pictures + [pictures[-1]]
        )
        body = self.requests[-1][2]
        self.assertNotIn("assets", body)
        self.assertEqual(len(body["images"]), 4)
        self.assertTrue(
            all(image.startswith("data:image/jpeg;base64,") for image in body["images"])
        )
        self.assertEqual(body["state"]["omitted_images"], 2)
        self.assertEqual(state, {"message": "图片"})
        pictures[0].convert_to_base64.assert_not_awaited()
        self.assertNotIn("TypeSafe-Danger-Zone-Images", self.requests[-1][1])

    async def test_disabled_images_and_preparation_timeout(self):
        picture = jevtests.JevHTTPTests.image(self)
        self.client.enable_images = False
        await self.client.evaluate("text", self.questions, images=[picture])
        self.assertNotIn("images", self.requests[-1][2])
        picture.convert_to_base64.assert_not_awaited()
        self.client.enable_images = True
        self.client.timeout = 0.01

        async def slow():
            await asyncio.sleep(1)

        picture.convert_to_base64 = slow
        with self.assertRaises(asyncio.TimeoutError):
            await self.client.evaluate("text", self.questions, images=[picture])
        self.assertEqual(len(self.requests), 1)


class CloudflareDebounceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloudflare_failure_preserves_images_and_merged_text(self):
        from test_main import Event

        plugin = debounce.DebouncePlugin(
            None, {"classifier_backend": "cloudflare", "timeout_seconds": 0}
        )
        await plugin._load_classifier_async()
        plugin.classifier.client.evaluate = AsyncMock(
            side_effect=[{"complete": {"noul": 0.1}}, cf.CloudflareError("quota")]
        )
        picture = types.SimpleNamespace(type="Image", file="base64://synthetic")
        for n, text in enumerate(["看看这张图", "是什么"]):
            event = Event(text, str(n))
            event.message_obj.message = [picture] if n == 0 else []
            request = types.SimpleNamespace(prompt=text, image_urls=[])
            await plugin.on_waiting_llm_request(event)
            await plugin.on_llm_request(event, request)
        self.assertFalse(event.stopped)
        self.assertEqual(request.prompt, "看看这张图 是什么")
        self.assertEqual(request.image_urls, [picture.file])
        self.assertEqual(
            plugin.classifier.client.evaluate.call_args.kwargs["images"], [picture]
        )
        await plugin.terminate()

    async def test_onnx_keeps_existing_lazy_load_and_two_fragment_shortcut(self):
        from test_main import Event

        plugin = debounce.DebouncePlugin(None, {"timeout_seconds": 0})
        fake = types.SimpleNamespace(predict=AsyncMock(return_value=(0.1, 0.1)))
        with tempfile.TemporaryDirectory() as directory:
            plugin.data_dir = Path(directory)
            model = plugin.data_dir / "models" / "small"
            (model / "tokenizer").mkdir(parents=True)
            (model / "model.onnx").touch()
            with patch.object(
                debounce, "SentenceClassifier", return_value=fake
            ) as factory:
                await plugin._load_classifier_async()
            factory.assert_called_once()
            for n, text in enumerate(["如果明天", "我们出去"]):
                event = Event(text, str(n))
                request = types.SimpleNamespace(prompt=text, image_urls=[])
                await plugin.on_waiting_llm_request(event)
                await plugin.on_llm_request(event, request)
            self.assertFalse(event.stopped)
            self.assertEqual(request.prompt, "如果明天 我们出去")
            fake.predict.assert_awaited_once_with("如果明天")
        await plugin.terminate()

    async def test_remote_backend_merges_three_fragments_without_onnx(self):
        from test_main import Event

        plugin = debounce.DebouncePlugin(
            None, {"classifier_backend": "cloudflare", "timeout_seconds": 0}
        )
        plugin._download_model_from_modelscope = AsyncMock(side_effect=AssertionError)
        await plugin._load_classifier_async()
        client = plugin.classifier.client
        client.evaluate = AsyncMock(
            side_effect=[{"complete": {"noul": p}} for p in [0.1, 0.2, 0.95]]
        )
        for n, text in enumerate(["如果明天", "不下雨", "我们去爬山"]):
            event = Event(text, str(n))
            request = types.SimpleNamespace(prompt=text, image_urls=[])
            await plugin.on_waiting_llm_request(event)
            await plugin.on_llm_request(event, request)
            self.assertEqual(event.stopped, n < 2)
        self.assertEqual(request.prompt, "如果明天 不下雨 我们去爬山")
        self.assertEqual(client.evaluate.await_count, 3)
        plugin._download_model_from_modelscope.assert_not_called()
        await plugin.terminate()
