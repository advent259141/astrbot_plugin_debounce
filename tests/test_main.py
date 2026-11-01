import asyncio
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def load_plugin():
    modules = {
        name: types.ModuleType(name)
        for name in [
            "astrbot",
            "astrbot.api",
            "astrbot.api.star",
            "astrbot.api.event",
            "astrbot.api.provider",
        ]
    }
    api = modules["astrbot.api"]
    api.logger = types.SimpleNamespace(
        **{name: lambda *_a, **_k: None for name in ["debug", "warning", "error"]}
    )
    api.AstrBotConfig = dict
    star = modules["astrbot.api.star"]

    class Star:
        def __init__(self, context):
            self.context = context

    star.Star = Star
    star.Context = object
    star.StarTools = types.SimpleNamespace(
        get_data_dir=lambda: Path(tempfile.gettempdir())
    )
    event = modules["astrbot.api.event"]
    event.AstrMessageEvent = object
    event.filter = types.SimpleNamespace(
        **{
            name: lambda *_a, **_k: lambda func: func
            for name in [
                "on_astrbot_loaded",
                "on_waiting_llm_request",
                "on_llm_request",
                "on_llm_response",
            ]
        }
    )
    modules["astrbot.api.provider"].ProviderRequest = object
    modules["astrbot.api.provider"].LLMResponse = object
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "debounce_test_module", root / "main.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


debounce = load_plugin()


class Event:
    def __init__(self, text, message_id="1"):
        self.message_str = text
        self.message_obj = types.SimpleNamespace(
            session_id="test", message_id=message_id
        )
        self.stopped = False

    def is_private_chat(self):
        return True

    def stop_event(self):
        self.stopped = True


class JevDebounceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = debounce.DebouncePlugin(
            None, {"classifier_backend": "jev", "timeout_seconds": 0}
        )
        await self.plugin._load_classifier_async()

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def submit(self, text, message_id, images=None):
        event = Event(text, message_id)
        event.message_obj.message = images or []
        request = types.SimpleNamespace(prompt=text, image_urls=[])
        await self.plugin.on_waiting_llm_request(event)
        await self.plugin.on_llm_request(event, request)
        return event, request

    async def test_jev_loading_does_not_load_or_download_onnx(self):
        self.plugin.classifier = None
        self.plugin._download_model_from_modelscope = AsyncMock(
            side_effect=AssertionError
        )
        with patch.dict(
            sys.modules,
            {
                name: None
                for name in ["onnxruntime", "transformers", "numpy", "modelscope"]
            },
        ):
            await self.plugin._load_classifier_async()
        self.assertIsInstance(self.plugin.classifier, debounce.JevSentenceClassifier)
        self.plugin._download_model_from_modelscope.assert_not_called()

    async def test_incomplete_fragments_keep_merging_until_complete(self):
        client = self.plugin.classifier.client
        client.evaluate = AsyncMock(
            side_effect=[
                {"complete": {"noul": 0.1}},
                {"complete": {"noul": 0.2}},
                {"complete": {"noul": 0.95}},
            ]
        )
        first, _ = await self.submit("如果明天", "1")
        second, _ = await self.submit("不下雨", "2")
        third, request = await self.submit("我们去爬山吧", "3")
        self.assertTrue(first.stopped)
        self.assertTrue(second.stopped)
        self.assertFalse(third.stopped)
        self.assertEqual(request.prompt, "如果明天 不下雨 我们去爬山吧")
        self.assertEqual(client.evaluate.await_count, 3)
        self.assertFalse(
            self.plugin.monitor_tasks
        )  # timeout=0 disables automatic sending
        state, questions = client.evaluate.call_args.args
        self.assertEqual(state["message"], request.prompt)
        self.assertEqual(questions["complete"]["type"], "noul")
        await self.plugin.on_llm_response(
            third, types.SimpleNamespace(completion_text="好")
        )
        self.assertEqual(self.plugin.buffers["test"].messages, [])

    async def test_failure_submits_all_buffered_messages(self):
        self.plugin.classifier.client.evaluate = AsyncMock(
            side_effect=[{"complete": {"noul": 0.1}}, ValueError("JEV unavailable")]
        )
        await self.submit("如果", "1")
        event, request = await self.submit("下雨", "2")
        self.assertFalse(event.stopped)
        self.assertEqual(request.prompt, "如果 下雨")

    async def test_new_message_during_remote_judgment_defers_old_request(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_judgment(_text):
            started.set()
            await release.wait()
            return 0.99, 0.99

        self.plugin.classifier.predict = slow_judgment
        first_task = asyncio.create_task(self.submit("如果明天", "1"))
        await started.wait()
        second = Event("不下雨我们去爬山", "2")
        await self.plugin.on_waiting_llm_request(second)
        release.set()
        first, _ = await first_task
        self.assertTrue(first.stopped)
        request = types.SimpleNamespace(prompt=second.message_str, image_urls=[])
        await self.plugin.on_llm_request(second, request)
        self.assertFalse(second.stopped)
        self.assertEqual(request.prompt, "如果明天 不下雨我们去爬山")

    async def test_cancelled_monitor_does_not_remove_replacement(self):
        self.plugin.config["timeout_seconds"] = 10
        self.plugin.classifier.predict = AsyncMock(return_value=(0.1, 0.1))
        await self.submit("如果", "1")
        old = self.plugin.monitor_tasks["test"]
        await asyncio.sleep(0)
        await self.submit("明天", "2")
        new = self.plugin.monitor_tasks["test"]
        self.assertIsNot(old, new)
        await asyncio.gather(old, return_exceptions=True)
        self.assertIs(self.plugin.monitor_tasks["test"], new)

    async def test_bad_probability_falls_back_to_submission(self):
        self.plugin.classifier.client.evaluate = AsyncMock(
            return_value={"complete": {"noul": True}}
        )
        event, _ = await self.submit("hello", "1")
        self.assertFalse(event.stopped)

    async def test_pure_picture_is_judged_and_forwarded(self):
        picture = types.SimpleNamespace(type="Image", file="base64://first")
        client = self.plugin.classifier.client
        client.evaluate = AsyncMock(return_value={"complete": {"noul": 0.99}})
        event, request = await self.submit("", "1", [picture])
        self.assertFalse(event.stopped)
        self.assertEqual(request.image_urls, [picture.file])
        self.assertEqual(client.evaluate.call_args.kwargs["images"], [picture])

    async def test_images_from_all_fragments_reach_judge_and_reply(self):
        first = types.SimpleNamespace(type="Image", file="base64://first")
        second = types.SimpleNamespace(type="Image", file="base64://second")
        client = self.plugin.classifier.client
        client.evaluate = AsyncMock(
            side_effect=[
                {"complete": {"noul": 0.1}},
                {"complete": {"noul": 0.1}},
                {"complete": {"noul": 0.99}},
            ]
        )
        await self.submit("比较这两张图", "1", [first])
        await self.submit("", "2", [second])
        event, request = await self.submit("有什么不同？", "3")
        self.assertFalse(event.stopped)
        self.assertEqual(request.prompt, "比较这两张图 有什么不同？")
        self.assertEqual(request.image_urls, [first.file, second.file])
        self.assertEqual(client.evaluate.call_args.kwargs["images"], [first, second])
        await self.plugin.on_llm_response(
            event, types.SimpleNamespace(completion_text="不同")
        )
        self.assertEqual(self.plugin.buffers["test"].images, [])
        self.assertEqual(self.plugin.buffers["test"].image_urls, [])

    async def test_picture_service_failure_preserves_images(self):
        picture = types.SimpleNamespace(type="Image", file="base64://first")
        self.plugin.classifier.client.evaluate = AsyncMock(
            side_effect=ValueError("HTTP 400")
        )
        event, request = await self.submit("这张图是什么？", "1", [picture])
        self.assertFalse(event.stopped)
        self.assertEqual(request.image_urls, [picture.file])

    async def test_timeout_keeps_earlier_pictures_when_last_event_is_text(self):
        picture = types.SimpleNamespace(type="Image", file="base64://first")
        self.plugin.classifier.client.evaluate = AsyncMock(
            return_value={"complete": {"noul": 0.1}}
        )
        await self.submit("看这张图", "1", [picture])
        await self.submit("我想知道", "2")
        self.plugin._send_fake_event = AsyncMock()
        await self.plugin._monitor_session("test", 0)
        self.assertEqual(
            self.plugin._send_fake_event.call_args.kwargs["images"], [picture]
        )

    async def test_failed_replay_keeps_text_and_pictures_buffered(self):
        picture = types.SimpleNamespace(type="Image", file="base64://first")
        self.plugin.classifier.client.evaluate = AsyncMock(
            return_value={"complete": {"noul": 0.1}}
        )
        await self.submit("看这张图", "1", [picture])
        self.plugin._send_fake_event = AsyncMock(return_value=False)
        await self.plugin._monitor_session("test", 0)
        self.assertEqual(self.plugin.buffers["test"].images, [picture])
        self.assertEqual(self.plugin.buffers["test"].get_full_text(), "看这张图")
        self.assertIn("test", self.plugin.waiting_sessions)

    async def test_disabled_plugin_does_not_call_jev(self):
        self.plugin.config["enabled"] = False
        self.plugin.classifier.predict = AsyncMock()
        await self.submit("hello", "1")
        self.plugin.classifier.predict.assert_not_awaited()

    async def test_disabled_cancellation_preserves_reply_and_next_message(self):
        self.plugin.config["cancel_on_new_message"] = False
        self.plugin.classifier.predict = AsyncMock(return_value=(0.99, 0.99))
        first, _ = await self.submit("第一个问题", "1")
        second = Event("第二个问题", "2")
        await self.plugin.on_waiting_llm_request(second)

        response = types.SimpleNamespace(completion_text="第一个回答")
        await self.plugin.on_llm_response(first, response)
        self.assertEqual(response.completion_text, "第一个回答")
        request = types.SimpleNamespace(prompt=second.message_str, image_urls=[])
        await self.plugin.on_llm_request(second, request)
        self.assertFalse(second.stopped)
        self.assertEqual(request.prompt, "第二个问题")
        self.assertEqual(self.plugin.buffers["test"].messages, ["第二个问题"])
        self.assertFalse(self.plugin.discard_next_response)
        self.assertFalse(self.plugin.should_cancel_msg_ids)

    async def test_disabled_cancellation_preserves_messages_during_judgment(self):
        self.plugin.config["cancel_on_new_message"] = False
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_judgment(_text):
            started.set()
            await release.wait()
            return 0.99, 0.99

        self.plugin.classifier.predict = slow_judgment
        first_task = asyncio.create_task(self.submit("第一个问题", "1"))
        await started.wait()
        queued = [Event("第二个问题", "2"), Event("第三个问题", "3")]
        for event in queued:
            await self.plugin.on_waiting_llm_request(event)
        release.set()
        first, _ = await first_task
        self.assertFalse(first.stopped)
        await self.plugin.on_llm_response(
            first, types.SimpleNamespace(completion_text="第一个回答")
        )

        # AstrBot runs request/response hooks inside the session lock.
        for event in queued:
            request = types.SimpleNamespace(prompt=event.message_str, image_urls=[])
            await self.plugin.on_llm_request(event, request)
            self.assertFalse(event.stopped)
            self.assertEqual(request.prompt, event.message_str)
            self.assertEqual(self.plugin.buffers["test"].messages, [event.message_str])
            response = types.SimpleNamespace(completion_text="回答")
            await self.plugin.on_llm_response(event, response)
            self.assertEqual(response.completion_text, "回答")
        self.assertFalse(self.plugin.waiting_msg_ids)
        self.assertFalse(self.plugin.should_cancel_msg_ids)

    async def test_disabled_cancellation_still_merges_incomplete_fragments(self):
        self.plugin.config["cancel_on_new_message"] = False
        self.plugin.config["timeout_seconds"] = 10
        self.plugin.classifier.predict = AsyncMock(
            side_effect=[(0.1, 0.1), (0.99, 0.99)]
        )
        first, _ = await self.submit("如果明天", "1")
        monitor = self.plugin.monitor_tasks["test"]
        second, request = await self.submit("不下雨就去爬山", "2")
        self.assertTrue(first.stopped)
        self.assertFalse(second.stopped)
        self.assertEqual(request.prompt, "如果明天 不下雨就去爬山")
        await asyncio.gather(monitor, return_exceptions=True)
        self.assertFalse(self.plugin.monitor_tasks)

    async def test_default_cancellation_discards_old_reply_and_merges(self):
        self.plugin.classifier.predict = AsyncMock(return_value=(0.99, 0.99))
        picture = types.SimpleNamespace(type="Image", file="base64://first")
        first, _ = await self.submit("第一个问题", "1", [picture])
        second = Event("补充内容", "2")
        await self.plugin.on_waiting_llm_request(second)
        response = types.SimpleNamespace(completion_text="旧回答")
        await self.plugin.on_llm_response(first, response)
        self.assertEqual(response.completion_text, "")
        request = types.SimpleNamespace(prompt=second.message_str, image_urls=[])
        await self.plugin.on_llm_request(second, request)
        self.assertFalse(second.stopped)
        self.assertEqual(request.prompt, "第一个问题 补充内容")
        self.assertEqual(request.image_urls, [picture.file])
        self.assertEqual(self.plugin.buffers["test"].images, [picture])

    async def test_default_cancellation_merges_messages_waiting_for_lock(self):
        self.plugin.classifier.predict = AsyncMock(return_value=(0.99, 0.99))
        first = Event("第一段", "1")
        second = Event("第二段", "2")
        await self.plugin.on_waiting_llm_request(first)
        await self.plugin.on_waiting_llm_request(second)
        first_request = types.SimpleNamespace(prompt=first.message_str, image_urls=[])
        await self.plugin.on_llm_request(first, first_request)
        self.assertTrue(first.stopped)
        second_request = types.SimpleNamespace(prompt=second.message_str, image_urls=[])
        await self.plugin.on_llm_request(second, second_request)
        self.assertFalse(second.stopped)
        self.assertEqual(second_request.prompt, "第一段 第二段")

    async def test_timeout_flushes_and_unload_cancels_monitors(self):
        self.plugin.config["timeout_seconds"] = 10
        self.plugin.classifier.predict = AsyncMock(return_value=(0.1, 0.1))
        await self.submit("如果", "1")
        task = self.plugin.monitor_tasks["test"]
        self.plugin._send_fake_event = AsyncMock()
        await self.plugin._monitor_session("test", 0)
        self.plugin._send_fake_event.assert_awaited_once()
        # The manual monitor is only used to exercise timeout behavior.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self.submit("然后", "2")
        active = list(self.plugin.monitor_tasks.values())
        await self.plugin.terminate()
        self.assertTrue(all(task.done() for task in active))
