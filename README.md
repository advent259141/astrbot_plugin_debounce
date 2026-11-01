# AstrBot 消息防抖插件

> ⚠️ **注意：本插件仅支持 AstrBot 4.11+ 版本使用**
> 
<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/advent259141/astrbot_plugin_debounce?style=flat-square)](https://github.com/yourusername/astrbot_plugin_debounce)
[![GitHub license](https://img.shields.io/github/license/advent259141/astrbot_plugin_debounce?style=flat-square)](LICENSE)
[![AstrBot](https://img.shields.io/badge/AstrBot-v4.11+-blue?style=flat-square)](https://github.com/AstrBotdevs/AstrBot)

**使用 BERT 模型智能判断用户是否说完一句话，减少不必要的 LLM 调用**

</div>

---

## 📖 简介

在实际使用聊天机器人时，用户经常会分多次发送一句话：

```
用户: 如果明天不下雨
用户: 我们去爬山吧
```

传统的聊天机器人会分别处理这两句话，导致：
- ❌ 第一句话语义不完整，LLM 回复质量差
- ❌ 浪费 API 调用次数和费用
- ❌ 用户体验不佳

本插件通过训练的 BERT 模型**实时判断句子完整性**，只在用户说完整句话后才发送给 LLM，实现：
- ✅ 智能防抖，自动合并未完成的消息
- ✅ 减少 LLM 调用次数，节省成本
- ✅ 提升对话质量和用户体验

---

## 🚀 功能特性

- **智能判断**：基于 BERT 模型，准确识别中文句子是否完整
- **双模型支持**：提供 Small（快速）和 Normal（精准）两种模型
- **自动下载**：首次使用时自动从 ModelScope 下载模型
- **灵活配置**：可调节判断阈值、超时时间等参数
- **零感知**：静默工作，不打扰用户
- **低开销**：使用 ONNX 推理，CPU 友好
- **远程判断**：支持 TypeSafe JEV（结构化决策模型）和 Cloudflare Workers AI（云端模型服务）
- **连接测试**：在插件可视化视图中点击按钮，验证已保存的远程模型配置

---

## 📦 安装

### 方法一：通过 AstrBot 插件市场（推荐）

1. 打开 AstrBot WebUI
2. 进入「插件管理」
3. 搜索「消息防抖」
4. 点击「安装」

### 方法二：手动安装

```bash
cd AstrBot/data/plugins
git clone https://github.com/yourusername/astrbot_plugin_debounce.git
```

然后在 AstrBot WebUI 中重载插件列表。

---

## ⚙️ 配置说明

### 使用 Cloudflare Workers AI（云端模型服务）

将 `classifier_backend`（判断后端）设为 `cloudflare`，填写以下配置，保存并重载插件：

| 配置项 | 中文用途 | 填写说明 |
|--------|----------|----------|
| `cloudflare_base_url` | Cloudflare 接口基地址 | `https://api.cloudflare.com/client/v4/accounts/你的账户ID/ai/run`，不需要追加模型名称 |
| `cloudflare_api_key` | Cloudflare 接口密钥 | 使用具有该账户 Workers AI 调用权限的 API Token（接口令牌） |
| `cloudflare_model` | Cloudflare 决策模型 | `@cf/cloudflare/clef-flash`（快速版，默认）、`@cf/cloudflare/clef`（标准版）或 `@cf/cloudflare/clef-omni`（多模态版），或选择“自定义” |
| `cloudflare_custom_model` | Cloudflare 自定义决策模型 | 选择“自定义”后填写完整模型 ID；仅支持兼容 Clef 决策接口的模型，普通聊天模型不能直接使用 |
| `cloudflare_timeout_seconds` | Cloudflare 请求超时秒数 | 单次远程判断的超时，与等待用户补充消息的 `timeout_seconds`（防抖等待秒数）独立 |
| `cloudflare_enable_images` | 让图片参与消息完整性判断（可选） | 默认关闭。开启会把图片发给 Cloudflare，帮助判断消息是否说完；不控制最终回复模型能否看图，也不会删除原图 |
| `cloudflare_max_images` | Cloudflare 单次判断图片上限 | 默认 4 张，范围 1–4；只限制判断请求，不删除交给回复模型的图片 |

插件使用 [Cloudflare Clef 原生决策接口](https://developers.cloudflare.com/workers-ai/models/clef-flash/)，
不能将 Cloudflare 地址填入 `jev_base_url`（JEV 接口基地址）。此后端不加载或下载本地 ONNX 模型。
接口异常时，插件会放行已经合并的消息，避免阻塞正常回复。
图片判断开启时，连续图文消息会结合缓存中的图片与文字判断；关闭后，纯图片直接交给回复模型，
图文消息仅按文字判断。JEV 与 Cloudflare 的图片开关和数量上限分别配置。

Cloudflare 免费计划的每日额度由账户共享；测试连接也会消耗少量额度。付费计划超额可能计费，
具体以 [Cloudflare 官方价格说明](https://developers.cloudflare.com/workers-ai/platform/pricing/)和账户设置为准。
密钥在配置页面遮罩显示，但仍会保存在服务器配置文件中，请勿将其放入截图、日志或公开反馈。

### 测试远程模型连接

1. 在插件配置中填写 Cloudflare 或 JEV 参数，保存并重载插件。
2. 在 AstrBot WebUI（管理面板）的插件列表中点击「消息防抖」卡片进入详情，打开「连接测试」视图。
3. 点击「刷新配置」，核对当前判断后端和模型，再点击「测试连接」。

测试只调用一次当前已保存并加载的配置，发送固定短测试句，检查接口连接及完整性判断结果的格式，
不读取聊天记录、不进入消息缓存、不触发聊天回复。执行时按钮会暂时禁用，完成后显示成功或失败及耗时。
修改配置后需要先保存并重载，再刷新测试页；页面无法读取配置弹窗中未保存的输入。
ONNX（本地判断模型）无需测试远程连接。

按钮使用 AstrBot 官方插件可视化视图（Views），已确认 AstrBot v4.29.3 支持。
旧版 AstrBot 若没有视图入口，仍可使用插件原有消息防抖功能；不能在普通配置弹窗中显示此按钮。
若页面提示读取失败，请检查管理面板登录状态；认证、额度、网络或返回格式错误则按页面提示检查服务配置。
连接成功只证明接口与结果格式可用，不保证所有中文消息的语义判断都准确。

### 使用 TypeSafe JEV

将 `classifier_backend` 切换为 `jev`，填写 `jev_api_key`，然后重载插件。
也可将密钥放在 AstrBot 进程的环境变量 `TYPESAFE_API_KEY` 中，配置中的密钥优先。

```json
{
  "enabled": true,
  "classifier_backend": "jev",
  "jev_api_key": "你的 TypeSafe API Key",
  "jev_base_url": "https://api.typesafe.ai/v1",
  "jev_model": "jev-latest",
  "jev_timeout_seconds": 10,
  "send_threshold": 0.8,
  "timeout_seconds": 10
}
```

使用[官方 System One 接口](https://docs.typesafe.ai/api/system-one)的 `Noul`
问题获取“用户是否已表达完整”的概率，与 `send_threshold` 比较。
连续收到多条片段时会重新判断合并后的完整文本，不会因为已收到两条就直接提交。
JEV 模式不加载 ONNX、Tokenizer，也不下载 ModelScope 模型；仅需 AstrBot 自带的
`aiohttp` 即可运行。`requirements.txt` 仍保留原有 ONNX 后端的依赖，常规安装会安装这些依赖。
默认后端仍为 `onnx`，`model_type` 仅对 ONNX 生效。

JEV 超时、缺少密钥、HTTP 错误或返回值非法时，直接提交已合并的消息，避免缓存丢失或阻塞回复。
`timeout_seconds` 是等待用户后续消息的时长，`jev_timeout_seconds` 是单次远程判断的超时，
两者相互独立；前者设为 0 时不自动发送。`jev_base_url` 支持 `/v1` 基地址或完整的
`/v1/systemone` 地址，不是 Chat Completions 接口。修改后端、模型或密钥后需重载插件。

### 使用 OpenRouter JEV

设置 `jev_base_url` 为 `https://openrouter.ai/api/v1`，`jev_model` 为
`typesafe/jev-1.14`，`jev_api_key` 填 OpenRouter 密钥，然后重载插件。
文本与图片均支持；插件会将 OpenRouter 图片内嵌到 `state.images` 的 `image_url` 中，
与完整文字上下文一起发送。OpenRouter 会忽略原生 `assets`，因此自动使用此格式。

### 图片与图文消息

JEV 后端默认启用 `jev_enable_images`，纯图片也会参与完整性判断。
连续图文消息合并时，JEV 会结合缓存中的图片与文字判断；传给回复模型的
`image_urls` 会保留各条消息的图片，超时补发也会保留早先消息的图片。
`jev_max_images` 默认 4（范围 1–8），限制单次传给 JEV 的最近图片数量，
不会删除传给回复模型的其他图片。关闭图片判断时，纯图片直接交给回复模型，图文仅按文字判断。

TypeSafe 官方及原生中转使用[原生实验性图片接口](https://docs.typesafe.ai/images)的 `assets` 和
`TypeSafe-Danger-Zone-Images: true` 请求头。图片会校正方向、取动画首帧、
转换 JPEG 并缩放/留白至官方尺寸范围，原图不变。中转服务必须支持这些字段和请求头。
图片读取失败或接口拒绝图片时，直接提交已合并的原始图片和文字。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model_type` | 选择 | `small` | 模型类型：`small`（轻量快速）/ `normal`（精准度高） |
| `send_threshold` | 浮点 | `0.8` | 完整性判断阈值（0-1），越高越严格 |
| `timeout_seconds` | 整数 | `10` | 超时自动发送时间（秒），设为 0 禁用 |
| `enabled` | 布尔 | `true` | 是否启用插件 |
| `cancel_on_new_message` | 布尔 | `true` | LLM 回复前收到新消息时取消回复 |
| `debug_mode` | 布尔 | `false` | 调试模式，输出详细日志 |

### 配置示例

```json
{
  "model_type": "small",
  "send_threshold": 0.6,
  "timeout_seconds": 30,
  "enabled": true,
  "cancel_on_new_message": true,
  "debug_mode": false
}
```

---

## 💡 使用示例

### 场景 1：分段输入

```
用户: 如果明天不下雨
      ↓ 插件判断：未完整，等待...
      
用户: 我们去爬山吧
      ↓ 插件判断：完整，发送给 LLM
      
LLM:  好的！如果明天天气好，我们可以一起去爬山~
```

### 场景 2：用户补充内容（启用 cancel_on_new_message）

```
用户: 如果明天不下雨
      ↓ 判断：未完整，缓存

用户: 我们去爬山吧
      ↓ 判断：完整，发送给 LLM
      ⏳ LLM 正在思考中...

用户: 顺便带上野餐垫
      ↓ 检测到：用户在等待回复时又发消息
      ❌ 取消 LLM 当前回复
      ✅ 将新消息加入缓存，继续等待

用户: 和水果
      ↓ 判断：完整，合并所有内容发送
      
LLM:  好的！明天天气好的话，我们去爬山，我会准备野餐垫和水果~
```

### 场景 3：超时处理

```
用户: 虽然今天很累
      ↓ 等待 30 秒...
      
      ↓ 超时，自动发送
      
LLM:  虽然今天很累，但还是要注意休息哦！
```

---

## 🔧 模型说明

插件提供两种模型，均托管在 [ModelScope](https://modelscope.cn/)：

| 模型 | 大小 | 速度 | 准确率 | 推荐场景 |
|------|------|------|--------|---------|
| **Small** | ~10MB | ⚡⚡⚡ | ~90% | 日常使用，CPU 环境 |
| **Normal** | ~100MB | ⚡⚡ | ~95% | 高准确度需求 |

### 模型仓库

- Small: [advent259141/astrbot_debouncer_small](https://modelscope.cn/models/advent259141/astrbot_debouncer_small)
- Normal: [advent259141/astrbot_debouncer_normal](https://modelscope.cn/models/advent259141/astrbot_debouncer_normal)

首次使用会自动下载，无需手动操作。

---

## 📊 训练数据

模型基于以下类型的中文句子训练：

| 标签 | 说明 | 示例 |
|------|------|------|
| **0 (WAIT)** | 句子未完整 | `"如果明天不下雨"` |
| **1 (SEND)** | 句子已完整 | `"如果明天不下雨我们去爬山"` |

涵盖常见的复句结构：
- 条件句：`如果...`、`假如...`
- 转折句：`虽然...`、`尽管...`
- 因果句：`因为...`、`由于...`
- 递进句：`其实...`、`也就是说...`

---

## 🐛 调试

开启调试模式后，日志会输出详细信息：

```
[防抖] 文本: '如果明天不下雨' | 完整概率: 0.2341 | 判定: 等待
[防抖] 文本: '如果明天不下雨 我们去爬山' | 完整概率: 0.8762 | 判定: 发送
[防抖] 完整发送: 如果明天不下雨 我们去爬山
```

---

## ❓ 常见问题

### Q1: 插件会拦截其他插件的 LLM 调用吗？

**A**: 不会。插件只拦截用户发送消息触发的 LLM 请求，其他插件通过 `context.llm_generate()` 直接调用 LLM 不受影响。

### Q2: 模型下载失败怎么办？

**A**: 
1. 检查网络连接
2. 手动下载模型文件放入 `models/` 目录
3. 或切换到 `small` 模型（体积更小）

### Q3: 如何调整判断的严格程度？

**A**: 修改 `send_threshold`：
- 值越大越严格（需要更高的完整概率）
- 建议范围：0.8 - 0.9

### Q4: 插件会影响性能吗？

**A**: 影响极小。ONNX 推理在 CPU 上耗时 < 10ms，可忽略不计。

### Q5: 什么是"取消回复"功能？

**A**: 当用户在等待 LLM 回复时又发送新消息，插件会：
- **启用时**：自动取消当前 LLM 回复，合并新消息后重新发送
- **禁用时**：不取消回复，新消息会作为独立消息处理

建议保持启用，以获得更好的对话体验。

---

## 🔄 取消消息机制详解

当启用 `cancel_on_new_message` 时，插件会在 LLM 处理期间检测新消息，并智能取消过时的回复。

### 工作原理

#### 核心钩子

插件使用三个 AstrBot 事件钩子协同工作：

| 钩子 | 执行时机 | 作用 |
|------|----------|------|
| `on_waiting_llm_request` | session lock **之前** | 检测新消息到达，标记旧响应需丢弃 |
| `on_llm_request` | session lock **之后** | BERT 判断完整性，管理缓冲区 |
| `on_llm_response` | LLM 响应返回后 | 检查是否需要丢弃响应 |

#### 关键时序

```
┌─────────────────────────────────────────────────────────────────┐
│  消息1: "小面包小面包"                                            │
├─────────────────────────────────────────────────────────────────┤
│  T1: on_waiting_llm_request(消息1)                                  │
│      → 无旧任务，跳过                                             │
│                                                                 │
│  T2: on_llm_request(消息1)                                       │
│      → BERT 判断: 0.04 (未完整)                                   │
│      → buffer = ["小面包小面包"]                                  │
│      → event.stop_event() 阻止发送                               │
│      → 启动 30秒 监控任务                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 3秒后
┌─────────────────────────────────────────────────────────────────┐
│  消息2: "我想你了"                                               │
├─────────────────────────────────────────────────────────────────┤
│  T3: on_waiting_llm_request(消息2)                                  │
│      → 取消监控任务 ✅                                            │
│                                                                 │
│  T4: on_llm_request(消息2)                                       │
│      → 检测到 waiting_sessions 有消息1                            │
│      → buffer = ["小面包小面包", "我想你了"]                       │
│      → BERT 判断: 0.97 (完整)                                     │
│      → req.prompt = "小面包小面包 我想你了"                        │
│      → pending_llm_sessions["session"] = "小面包小面包 我想你了"   │
│      → 发送 LLM 请求 🚀                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 2秒后 (LLM 还在处理中)
┌─────────────────────────────────────────────────────────────────┐
│  消息3: "想和你聊聊天"                                           │
├─────────────────────────────────────────────────────────────────┤
│  T5: on_will_llm_request(消息3) ⚡ 在 session lock 之前执行！      │
│      → 检测到 pending_llm_sessions 有正在处理的请求               │
│      → discard_responses.add("session") 标记丢弃 🎯              │
│      → 恢复 "小面包小面包 我想你了" 到 buffer                      │
│      → buffer = ["小面包小面包 我想你了"]                          │
│                                                                 │
│  T6: [被 session lock 阻塞，等待消息2的LLM完成...]                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 8秒后 (LLM 处理完成)
┌─────────────────────────────────────────────────────────────────┐
│  消息2 的 LLM 响应返回                                           │
├─────────────────────────────────────────────────────────────────┤
│  T7: on_llm_response(消息2的响应)                                 │
│      → 检测到 discard_responses 包含此会话                        │
│      → event.stop_event() 丢弃响应 🚫                            │
│      → 用户看不到这个过时的回复 ✅                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ session lock 释放
┌─────────────────────────────────────────────────────────────────┐
│  消息3 继续处理                                                  │
├─────────────────────────────────────────────────────────────────┤
│  T8: on_llm_request(消息3)                                       │
│      → buffer = ["小面包小面包 我想你了", "想和你聊聊天"]          │
│      → BERT 判断: 0.95 (完整)                                     │
│      → req.prompt = "小面包小面包 我想你了 想和你聊聊天"           │
│      → 发送 LLM 请求 🚀                                           │
│                                                                 │
│  T9: on_llm_response(消息3的响应)                                 │
│      → 正常返回 ✅                                                │
│      → 用户看到完整的回复 🎉                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么需要 `on_waiting_llm_request`？

AstrBot 使用 **session lock** 防止同一会话的并发 LLM 请求。这意味着：

```
消息2 正在调用 LLM
    ↓
消息3 到达
    ↓
消息3 的 on_llm_request 被 session lock 阻塞
    ↓
等待消息2 的 LLM 完成后才能执行
    ↓
此时检测"新消息"已经太晚了！
```

**解决方案**：`on_waiting_llm_request` 钩子在 **session lock 之前** 执行，让我们能在第一时间检测到新消息并标记旧响应需要丢弃。

### 状态管理

| 状态集合 | 类型 | 作用 |
|----------|------|------|
| `buffers` | `Dict[session_id, MessageBuffer]` | 存储未完成的消息 |
| `waiting_sessions` | `Set[session_id]` | 标记正在等待更多消息的会话 |
| `pending_llm_sessions` | `Dict[session_id, message_text]` | 记录正在处理 LLM 的会话及其消息 |
| `discard_responses` | `Set[session_id]` | 标记需要丢弃响应的会话 |
| `monitor_tasks` | `Dict[session_id, Task]` | 超时监控任务 |
| `skip_debounce_msg_ids` | `Set[msg_id]` | 跳过防抖的伪造消息 ID |

### 消息恢复机制

当旧响应被取消时，插件会将原消息内容恢复到 buffer：

```python
# 在 on_waiting_llm_request 中
if session_id in self.pending_llm_sessions:
    old_message = self.pending_llm_sessions[session_id]  # "小面包 我想你了"
    self.discard_responses.add(session_id)
    
    # 恢复到 buffer，与新消息合并
    buffer.messages.insert(0, old_message)
    # buffer 变为 ["小面包 我想你了", "想和你聊聊天"]
```

这确保用户发送的所有内容都不会丢失。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！



## 📄 许可证

[Apache License 2.0](LICENSE)



**如果这个插件对你有帮助，欢迎给个 ⭐ Star！**

</div>




