"""
AstrBot 消息防抖插件
使用 BERT 模型判断用户是否说完一句话，避免频繁调用 LLM
"""

import os
import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.star import StarTools
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger, AstrBotConfig


@dataclass
class MessageBuffer:
    """消息缓冲区，用于存储用户未完成的消息"""
    messages: list = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    event: Optional[AstrMessageEvent] = None  # 保存最后一个event用于超时发送
    
    def add(self, message: str, event: AstrMessageEvent = None):
        self.messages.append(message)
        self.last_update = time.time()
        if event:
            self.event = event
    
    def get_full_text(self) -> str:
        return " ".join(self.messages)
    
    def clear(self):
        self.messages = []
        self.last_update = time.time()
        self.event = None
    
    def is_timeout(self, timeout_seconds: int) -> bool:
        if timeout_seconds <= 0:
            return False
        return time.time() - self.last_update > timeout_seconds


class SentenceClassifier:
    """句子完整性分类器"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(model_path)
        logger.debug(f"🚀 消息防抖模型已加载: {model_path}")
    
    def _predict_sync(self, text: str) -> tuple[float, float]:
        """
        同步预测（内部使用）
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            padding=True, 
            truncation=True,
            max_length=64
        )
        
        outputs = self.session.run(
            output_names=["logits"],
            input_feed={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )
        
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定的 softmax
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        score_send = float(softmax_probs[0][1])  # Label 1 = SEND
        return score_send, score_send
    
    async def predict(self, text: str) -> tuple[float, float]:
        """
        异步预测句子是否完整（使用线程池避免阻塞事件循环）
        返回: (完整概率, 完整概率)
        """
        return await asyncio.to_thread(self._predict_sync, text)



class DebouncePlugin(Star):
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        logger.debug(f"插件配置: {dict(config)}")
        
        # 消息缓冲区 {session_id: MessageBuffer}
        self.buffers: Dict[str, MessageBuffer] = {}
        
        # 正在等待中的会话集合（等待更多消息）
        self.waiting_sessions: set = set()
        
        # 后台监控任务集合 {session_id: Task}
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        
        # 跳过防抖的消息ID集合（伪造的消息）
        self.skip_debounce_msg_ids: set = set()
        
        # 正在处理LLM请求的会话集合
        self.pending_llm_sessions: set = set()
        
        # 需要丢弃下一个响应的会话集合（简化：只需要标记，不需要计数）
        self.discard_next_response: set = set()
        
        # 正在等待session lock的消息ID {session_id: msg_id}
        self.waiting_msg_ids: Dict[str, str] = {}
        
        # 应该被取消的消息ID集合（在on_llm_request中直接取消）
        self.should_cancel_msg_ids: set = set()
        
        # 分类器（延迟加载）
        self.classifier = None
        
        # 模型加载锁（防止并发加载）
        self._model_load_lock = asyncio.Lock()
        
        # 使用 StarTools 获取数据目录
        self.data_dir = StarTools.get_data_dir()
        
        # 超时检查任务
        self._timeout_task: Optional[asyncio.Task] = None
    
    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 加载完成后的初始化"""
        # 启动后台清理任务
        self._timeout_task = asyncio.create_task(self._timeout_checker())
        logger.debug("消息防抖插件后台任务已启动")
        
        # 异步预加载模型
        try:
            await self._load_classifier_async()
        except Exception as e:
            logger.warning(f"消息防抖插件模型预加载失败: {e}")
            logger.warning("插件将在首次使用时尝试加载模型")
    
    async def _load_classifier_async(self):
        """异步加载分类模型"""
        async with self._model_load_lock:
            if self.classifier is not None:
                return
            
            model_type = self.config.get("model_type", "small")
            model_dir = self.data_dir / "models" / model_type
            model_path = model_dir / "model.onnx"
            tokenizer_path = model_dir / "tokenizer"
            
            # 检查模型是否存在，不存在则自动下载
            if not model_path.exists():
                logger.debug(f"模型文件不存在，尝试从 ModelScope 下载: {model_type}")
                success = await asyncio.to_thread(
                    self._download_model_from_modelscope, 
                    model_type, 
                    str(model_dir)
                )
                if not success:
                    logger.error(f"❌ 模型下载失败: {model_type}")
                    raise FileNotFoundError(f"模型文件不存在且下载失败: {model_path}")
            
            if not tokenizer_path.exists():
                logger.error(f"❌ Tokenizer 不存在: {tokenizer_path}")
                raise FileNotFoundError(f"Tokenizer 不存在: {tokenizer_path}")
            
            # 在线程池中加载模型（避免阻塞）
            self.classifier = await asyncio.to_thread(
                SentenceClassifier,
                str(model_path),
                str(tokenizer_path)
            )
            logger.debug(f"✅ 消息防抖插件已加载模型: {model_type}")
    
    def _download_model_from_modelscope(self, model_type: str, target_dir: str) -> bool:
        """从 ModelScope 下载模型（同步方法，应在线程池中调用）"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            # ModelScope 模型仓库映射
            model_repos = {
                "small": "advent259141/astrbot_debouncer_small",
                "normal": "advent259141/astrbot_debouncer_normal"
            }
            
            repo_id = model_repos.get(model_type)
            if not repo_id:
                logger.warning(f"模型类型 {model_type} 无需下载")
                return False
            
            logger.debug(f"🔄 正在从 ModelScope 下载模型: {repo_id}")
            
            # 下载到数据目录的 .cache
            cache_path = self.data_dir / ".cache"
            cache_dir = snapshot_download(
                repo_id,
                cache_dir=str(cache_path)
            )
            
            # 复制文件到目标目录
            import shutil
            os.makedirs(target_dir, exist_ok=True)
            
            # 复制模型文件
            src_model = os.path.join(cache_dir, "model", "model.onnx")
            if os.path.exists(src_model):
                shutil.copy2(src_model, os.path.join(target_dir, "model.onnx"))
                logger.debug("✅ 模型文件下载完成")
            
            # 复制 tokenizer 目录
            src_tokenizer = os.path.join(cache_dir, "tokenizer")
            dst_tokenizer = os.path.join(target_dir, "tokenizer")
            if os.path.exists(src_tokenizer):
                shutil.copytree(src_tokenizer, dst_tokenizer, dirs_exist_ok=True)
                logger.debug("✅ Tokenizer 文件下载完成")
            
            logger.debug(f"🎉 模型 {model_type} 下载成功")
            return True
            
        except ImportError:
            logger.error("❌ modelscope 库未安装，请运行: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"❌ 从 ModelScope 下载模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_buffer(self, session_id: str) -> MessageBuffer:
        """获取或创建消息缓冲区"""
        if session_id not in self.buffers:
            self.buffers[session_id] = MessageBuffer()
        return self.buffers[session_id]
    
    @filter.on_waiting_llm_request(priority=100)
    async def on_waiting_llm_request(self, event: AstrMessageEvent):
        """即将调用 LLM 时的通知（在 session lock 之前）- 用于检测新消息到达"""
        
        # 检查是否启用
        if not self.config.get("enabled", True):
            return
        
        # 检查使用场景
        usage_scope = self.config.get("usage_scope", "both")
        is_private = event.is_private_chat()
        if usage_scope == "group" and is_private:
            return
        if usage_scope == "private" and not is_private:
            return
        
        session_id = event.message_obj.session_id
        msg_id = event.message_obj.message_id
        
        # 跳过伪造消息
        if msg_id in self.skip_debounce_msg_ids:
            # 不在这里移除，留到 on_llm_request 移除
            logger.debug(f"[Debounce] 检测到伪造消息（跳过状态检查）: {session_id}")
            return
        
        # 如果该session有消息正在等待锁，取消它（只处理最新的消息）
        if session_id in self.waiting_msg_ids:
            old_msg_id = self.waiting_msg_ids[session_id]
            self.should_cancel_msg_ids.add(old_msg_id)
            logger.debug(f"[Debounce] 标记前一个等待中的消息应被取消: {session_id}, msg_id: {old_msg_id}")
        
        # 记录当前消息正在等待锁
        self.waiting_msg_ids[session_id] = msg_id
        
        # 取消之前的监控任务
        if session_id in self.monitor_tasks:
            self.monitor_tasks[session_id].cancel()
            del self.monitor_tasks[session_id]
            logger.debug(f"[Debounce] 新消息到达，取消监控任务: {session_id}")
        
        # 如果之前的请求还在处理中，标记需要丢弃其响应
        if session_id in self.pending_llm_sessions:
            self.discard_next_response.add(session_id)
            logger.debug(f"[Debounce] 新消息到达，标记旧LLM响应需要丢弃: {session_id}")
            
            # 恢复消息到buffer（保证内容完整）
            buffer = self._get_buffer(session_id)
            # 注意：不需要恢复消息内容，因为buffer中已经有等待的消息了
    
    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """核心防抖逻辑"""
        
        # 检查是否启用
        if not self.config.get("enabled", True):
            return
        
        # 检查使用场景
        usage_scope = self.config.get("usage_scope", "both")
        is_private = event.is_private_chat()
        if usage_scope == "group" and is_private:
            return
        if usage_scope == "private" and not is_private:
            return
        
        session_id = event.message_obj.session_id
        message_text = event.message_str.strip()
        msg_id = event.message_obj.message_id
        
        # 清除等待记录（当前消息获得了锁）
        if self.waiting_msg_ids.get(session_id) == msg_id:
            del self.waiting_msg_ids[session_id]
        
        # 优先检查：如果这条消息被标记为应该取消，加入buffer后取消
        if msg_id in self.should_cancel_msg_ids:
            self.should_cancel_msg_ids.remove(msg_id)
            if not message_text:
                event.stop_event()
                return
            # 将消息加入buffer，确保它能与后续消息合并
            buffer = self._get_buffer(session_id)
            buffer.add(message_text, event)
            # 标记为等待状态，以便后续消息能合并
            self.waiting_sessions.add(session_id)
            logger.debug(f"[Debounce] 消息已加入buffer但被取消（等待合并）: {session_id}, msg_id: {msg_id}")
            event.stop_event()
            return
        
        # 检查模型是否加载成功，如果未加载则尝试加载
        if self.classifier is None:
            try:
                await self._load_classifier_async()
            except Exception as e:
                logger.warning(f"模型加载失败，跳过防抖: {e}")
                return
        
        if self.classifier is None:
            return
        
        if not message_text:
            return
        
        # 跳过伪造消息（已经是超时后主动发送的，直接通过）
        if msg_id in self.skip_debounce_msg_ids:
            self.skip_debounce_msg_ids.remove(msg_id)
            # 标记正在处理LLM请求
            self.pending_llm_sessions.add(session_id)
            logger.debug(f"[Debounce] 伪造消息直接通过: {session_id}")
            return
        
        buffer = self._get_buffer(session_id)
        threshold = self.config.get("send_threshold", 0.5)
        timeout_seconds = self.config.get("timeout_seconds", 30)
        
        # 关键修复：如果该会话正在等待中 OR 有LLM正在处理，将新消息添加到buffer
        # 这样可以确保连续多条消息都能正确合并
        should_merge = (session_id in self.waiting_sessions) or (session_id in self.pending_llm_sessions)
        
        if should_merge:
            buffer.add(message_text, event)
            logger.debug(f"消息已添加到缓冲区: {session_id}, buffer消息数: {len(buffer.messages)}")
            
            # 优化：如果buffer中有多条消息（说明有被取消的消息），直接发送
            if len(buffer.messages) > 1:
                # 取消监控任务
                if session_id in self.monitor_tasks:
                    self.monitor_tasks[session_id].cancel()
                    del self.monitor_tasks[session_id]
                
                # 清除等待标记
                self.waiting_sessions.discard(session_id)
                
                # 直接发送合并消息
                full_text = buffer.get_full_text()
                req.prompt = full_text
                logger.debug(f"[Debounce] 多条消息合并直接发送（跳过判断）: {full_text}")
                
                # 注意：不清空buffer！等LLM成功响应后再清空
                # 因为在响应前可能还有新消息到达,需要全部合并
                
                # 注意：不清除discard标记！因为旧的LLM响应可能还在路上
                # 旧响应到达时会检测到discard标记并被丢弃
                
                # 标记该会话正在处理新的 LLM 请求
                self.pending_llm_sessions.add(session_id)
                return
            
            # 只有一条消息，需要判断完整性
            full_text = buffer.get_full_text()
            score_send, _ = await self.classifier.predict(full_text)
            is_complete = score_send >= threshold
            logger.debug(f"完整概率: {score_send:.2f} | 判定: {'发送' if is_complete else '继续等待'}")
            
            if is_complete:
                # 现在完整了，取消监控任务
                if session_id in self.monitor_tasks:
                    self.monitor_tasks[session_id].cancel()
                    del self.monitor_tasks[session_id]
                
                # 清除等待标记
                self.waiting_sessions.discard(session_id)
                
                # 修改 ProviderRequest 的 prompt 为合并后的完整文本
                req.prompt = full_text
                logger.debug(f"[Debounce] 合并消息发送: {full_text}")
                
                # 注意：不清空buffer！等LLM成功响应后再清空
                
                # 清除丢弃标记（这是新的请求，不应该被丢弃）
                self.discard_next_response.discard(session_id)
                
                # 标记该会话正在处理新的 LLM 请求
                self.pending_llm_sessions.add(session_id)
                return  # 让这条消息正常发送
            else:
                # 还是不完整，阻止当前消息
                event.stop_event()
                
                # 重新启动监控任务（因为之前的任务在 on_waiting_llm_request 中被取消了）
                if session_id not in self.monitor_tasks:
                    task = asyncio.create_task(self._monitor_session(session_id, timeout_seconds))
                    self.monitor_tasks[session_id] = task
                    logger.debug(f"[Debounce] 重新启动监控任务: {session_id}")
                return
        
        # 首次收到消息，添加到buffer
        buffer.add(message_text, event)
        full_text = buffer.get_full_text()
        
        # 判断完整性（异步调用）
        score_send, _ = await self.classifier.predict(full_text)
        is_complete = score_send >= threshold
        logger.debug(f"完整概率: {score_send:.2f} | 判定: {'发送' if is_complete else '等待'}")
        
        if is_complete:
            # 完整，让消息通过（不清空buffer,等LLM响应后清空）
            # 清除丢弃标记（这是新的请求，不应该被丢弃）
            self.discard_next_response.discard(session_id)
            # 标记该会话正在处理 LLM 请求
            self.pending_llm_sessions.add(session_id)
            return
        else:
            # 不完整，阻止发送，启动监控任务
            event.stop_event()
            self.waiting_sessions.add(session_id)
            
            # 启动后台监控任务（如果还没有）
            if session_id not in self.monitor_tasks:
                task = asyncio.create_task(self._monitor_session(session_id, timeout_seconds))
                self.monitor_tasks[session_id] = task
            return
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应后的钩子 - 用于丢弃过时的响应"""
        # 检查使用场景
        usage_scope = self.config.get("usage_scope", "both")
        is_private = event.is_private_chat()
        if usage_scope == "group" and is_private:
            return
        if usage_scope == "private" and not is_private:
            return
        
        session_id = event.message_obj.session_id
        
        # 检查是否需要丢弃这个回复
        if session_id in self.discard_next_response:
            logger.debug(f"[Debounce] 已丢弃过时的 LLM 回复: {session_id}")
            # 输出空文本
            resp.completion_text = ""
        else:
            # 只有成功的响应才清空buffer
            if session_id in self.buffers:
                self.buffers[session_id].clear()
                logger.debug(f"[Debounce] LLM响应成功,清空buffer: {session_id}")
        
        # 清除待处理标记(无论是否丢弃,这个LLM会话都已结束)
        self.pending_llm_sessions.discard(session_id)
        self.discard_next_response.discard(session_id)
    
    async def _timeout_checker(self):
        """后台任务：定期清理过期的缓冲区"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            
            timeout_seconds = self.config.get("timeout_seconds", 30)
            if timeout_seconds <= 0:
                continue
            
            # 清理过期的缓冲区（超过3倍超时时间）
            expired_sessions = []
            for session_id, buffer in self.buffers.items():
                if buffer.messages and buffer.is_timeout(timeout_seconds * 3):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.buffers[session_id]
                logger.debug(f"[防抖] 清理过期缓冲区: {session_id}")
    
    async def terminate(self):
        """插件卸载时的清理"""
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        
        self.buffers.clear()
        self.pending_llm_sessions.clear()
        self.discard_next_response.clear()
        self.skip_debounce_msg_ids.clear()
        self.waiting_sessions.clear()
        self.waiting_msg_ids.clear()
        self.should_cancel_msg_ids.clear()
        self.classifier = None
        logger.debug("🛑 消息防抖插件已卸载")
    
    async def _monitor_session(self, session_id: str, timeout_seconds: int):
        """监控会话超时，超时后伪造事件发送"""
        try:
            # 等待超时时间
            await asyncio.sleep(timeout_seconds)
            
            # 检查buffer是否还在等待（可能已被新消息触发发送）
            if session_id not in self.waiting_sessions:
                return
            
            buffer = self.buffers.get(session_id)
            if not buffer:
                return
            
            # 超时了，获取缓存的消息和event
            full_text = buffer.get_full_text()
            saved_event = buffer.event
            
            if not full_text or not saved_event:
                return
            
            logger.debug(f"[Debounce] 等待超时，伪造事件发送: {session_id}")
            
            # 清除等待状态
            self.waiting_sessions.discard(session_id)
            buffer.clear()
            
            # 伪造一个新消息事件
            await self._send_fake_event(saved_event, full_text)
        
        except asyncio.CancelledError:
            # 任务被取消（有新消息到达）
            logger.debug(f"[Debounce] 监控任务被取消: {session_id}")
        except Exception as e:
            logger.error(f"[Debounce] 超时发送失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 清理任务记录
            self.monitor_tasks.pop(session_id, None)
    
    async def _send_fake_event(self, original_event: AstrMessageEvent, message_text: str):
        """伪造一个消息事件并发送到EventBus"""
        try:
            from astrbot.core.star.star_tools import StarTools
            from astrbot.core.message.components import Plain
            
            # 保留原始消息的富文本组件(图片、表情等)
            # 只替换文本部分,保留其他富文本元素
            original_message = original_event.message_obj.message
            new_message_components = []
            
            # 遍历原始消息组件,保留非Plain类型的组件(如图片、表情等)
            for component in original_message:
                if not isinstance(component, Plain):
                    new_message_components.append(component)
            
            # 在开头添加合并后的文本
            new_message_components.insert(0, Plain(message_text))
            
            # 创建新消息对象
            new_message = await StarTools.create_message(
                type=str(original_event.message_obj.type.value),
                self_id=original_event.get_self_id(),
                session_id=original_event.session_id,
                sender=original_event.message_obj.sender,
                message=new_message_components,
                message_str=message_text,
                group_id=original_event.get_group_id() or "",
                message_id=original_event.message_obj.message_id,  # 伪造消息时保留真实ID
            )
            
            # 标记这个消息需要跳过防抖
            self.skip_debounce_msg_ids.add(new_message.message_id)
            
            # 伪造事件并提交
            await StarTools.create_event(
                abm=new_message,
                platform=original_event.get_platform_name(),
                is_wake=True
            )
            
            logger.debug(f"[Debounce] 已伪造事件发送: {message_text[:50]}")
            
        except Exception as e:
            logger.error(f"[Debounce] 伪造事件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
