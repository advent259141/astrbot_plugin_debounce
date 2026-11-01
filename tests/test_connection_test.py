import asyncio
import importlib
import os
import types
import unittest
from unittest.mock import AsyncMock, patch

from test_main import debounce


diagnostics = importlib.import_module("debounce_test_module.connection_test")
cf = importlib.import_module("debounce_test_module.cloudflare_client")


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "classifier_backend": "cloudflare",
            "cloudflare_api_key": "test-only-key",
        }
        self.service = diagnostics.ConnectionTest(self.config)

    async def test_single_call_uses_saved_configuration_and_valid_probability(self):
        classifier = types.SimpleNamespace(predict=AsyncMock(return_value=(0.42, 0.42)))
        with patch.object(
            diagnostics, "remote_classifier", return_value=classifier
        ) as factory:
            result = await self.service.run()
        factory.assert_called_once_with(self.config)
        classifier.predict.assert_awaited_once_with("你好，请问你能帮助我吗？")
        self.assertTrue(result["success"])
        self.assertEqual(result["probability"], 0.42)
        self.assertNotIn("test-only-key", str(result))

    async def test_failure_does_not_become_fail_open_success(self):
        for error, code in [
            (cf.CloudflareError("quota"), "quota"),
            (asyncio.TimeoutError(), "timeout"),
            (ValueError("JEV HTTP 401"), "authentication"),
            (ValueError("JEV HTTP 429"), "rate_limit"),
            (ValueError("Configure jev_api_key or TYPESAFE_API_KEY"), "missing_key"),
            (ValueError("JEV connection failed"), "network"),
            (
                RuntimeError("https://example.test/?key=test-only-key"),
                "invalid_response",
            ),
        ]:
            classifier = types.SimpleNamespace(predict=AsyncMock(side_effect=error))
            with patch.object(
                diagnostics, "remote_classifier", return_value=classifier
            ):
                result = await self.service.run()
            self.assertFalse(result["success"])
            self.assertEqual(result["code"], code)
            self.assertNotIn("test-only-key", str(result))
            self.assertNotIn("example.test", str(result))

    async def test_duplicate_call_does_not_queue_and_lock_recovers(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def predict(_text):
            started.set()
            await release.wait()
            return 0.8, 0.8

        classifier = types.SimpleNamespace(predict=AsyncMock(side_effect=predict))
        with patch.object(diagnostics, "remote_classifier", return_value=classifier):
            task = asyncio.create_task(self.service.run())
            await started.wait()
            self.assertTrue(self.service.info()["testing"])
            result = await self.service.run()
            self.assertEqual(result["code"], "busy")
            classifier.predict.assert_awaited_once()
            release.set()
            self.assertTrue((await task)["success"])
            self.assertFalse(self.service.info()["testing"])

    async def test_cancelled_test_releases_lock(self):
        started = asyncio.Event()

        async def predict(_text):
            started.set()
            await asyncio.sleep(10)

        with patch.object(
            diagnostics,
            "remote_classifier",
            return_value=types.SimpleNamespace(predict=predict),
        ):
            task = asyncio.create_task(self.service.run())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(self.service.lock.locked())

    async def test_local_backend_does_not_load_onnx(self):
        self.config["classifier_backend"] = "onnx"
        with patch.object(diagnostics, "remote_classifier", side_effect=AssertionError):
            result = await self.service.run()
        self.assertFalse(result["supported"])
        self.assertEqual(result["code"], "unsupported")

    async def test_model_metadata_cannot_echo_configured_or_environment_key(self):
        self.config["classifier_backend"] = "jev"
        for configured in ["synthetic-only-key", ""]:
            self.config["jev_api_key"] = configured
            self.config["jev_model"] = "prefix-synthetic-only-key"
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "synthetic-only-key"}):
                info = self.service.info()
            self.assertNotIn("synthetic-only-key", str(info))

    async def test_custom_model_metadata_is_resolved_and_sanitized(self):
        self.config.update(cloudflare_model="自定义", cloudflare_custom_model="@cf/cloudflare/clef-next")
        self.assertEqual(self.service.info()["model"], "@cf/cloudflare/clef-next")
        self.config["cloudflare_custom_model"] = "@cf/cloudflare/test-only-key"
        self.assertNotIn("test-only-key", str(self.service.info()))

    async def test_api_uses_web_contract_and_leaves_buffers_unchanged(self):
        routes = []
        context = types.SimpleNamespace(
            registered_web_apis=routes,
            register_web_api=lambda *args: routes.append(args),
        )
        web = types.ModuleType("astrbot.api.web")
        web.request = types.SimpleNamespace(method="GET")
        web.json_response = lambda data, **kwargs: (data, kwargs)
        with patch.dict("sys.modules", {"astrbot.api.web": web}):
            plugin = debounce.DebouncePlugin(context, self.config)
        route, handler, methods, _ = routes[0]
        self.assertEqual(route, "/astrbot_plugin_debounce/connection")
        self.assertEqual(methods, ["GET", "POST"])
        result, headers = await handler()
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("api_key", str(result))
        self.assertEqual(headers["headers"]["Cache-Control"], "no-store")
        plugin.buffers["untouched"] = object()
        snapshot = dict(plugin.buffers)
        web.request.method = "POST"
        plugin.connection_test.run = AsyncMock(return_value={"success": True})
        result, _ = await handler()
        self.assertTrue(result["data"]["success"])
        plugin.connection_test.run.assert_awaited_once_with()
        self.assertEqual(plugin.buffers, snapshot)
        await plugin.terminate()
        self.assertEqual(routes, [])

    async def test_old_astrbot_without_web_module_keeps_plugin_usable(self):
        context = types.SimpleNamespace(
            register_web_api=lambda *_: self.fail("registered unsupported API")
        )
        with patch.dict("sys.modules", {"astrbot.api.web": None}):
            plugin = debounce.DebouncePlugin(context, self.config)
        await plugin._load_classifier_async()
        self.assertIsInstance(plugin.classifier, debounce.SystemOneSentenceClassifier)
        await plugin.terminate()
