const bridge = window.AstrBotPluginView ?? window.AstrBotPluginPage;
const testButton = document.getElementById("test");
const refreshButton = document.getElementById("refresh");
const backendLabel = document.getElementById("backend");
const modelLabel = document.getElementById("model");
const note = document.getElementById("configuration-note");
const result = document.getElementById("result");
const resultTitle = document.getElementById("result-title");
const resultMessage = document.getElementById("result-message");
const resultDetail = document.getElementById("result-detail");

const backendNames = {
  cloudflare: "Cloudflare Workers AI（云端模型服务）",
  jev: "TypeSafe JEV（结构化决策模型）",
  onnx: "ONNX（本地判断模型）",
};
const errors = {
  missing_key: "缺少接口密钥，请填写并保存配置后重试。",
  configuration: "判断后端配置无效，请检查配置并重载插件。",
  invalid_url: "Cloudflare 接口地址无效，请填写官方 HTTPS 账户 ai/run 基地址。",
  invalid_model: "模型名称无效或不存在，请检查模型配置。",
  authentication: "接口密钥无效或已失效，请检查密钥。",
  permission: "无访问权限，请检查密钥权限与账户范围。",
  quota: "今日免费神经元额度已用完，请检查 Cloudflare 账户额度。",
  rate_limit: "请求受到限流，请稍后重试。",
  timeout: "连接超时，请检查服务器网络或请求超时设置。",
  network: "网络连接失败，请检查服务器的网络连接。",
  invalid_response: "模型返回格式异常，无法完成完整性判断。",
  invalid_request: "模型拒绝了请求，请检查模型及接口配置。",
  service: "模型服务暂时不可用，请稍后重试。",
  model_busy: "模型暂时繁忙，请稍后重试。",
  busy: "已有连接测试正在执行，请稍后刷新配置再试。",
  unsupported: "当前使用本地模型，不需要测试远程连接。",
};
let configuration = null;
let working = false;

function updateButtons() {
  testButton.disabled = working || !configuration?.supported || configuration.testing;
  refreshButton.disabled = working || !bridge;
}

function showResult(state, title, message, detail = "") {
  result.hidden = false;
  result.dataset.state = state;
  resultTitle.textContent = title;
  resultMessage.textContent = message;
  resultDetail.textContent = detail;
}

async function refreshConfiguration() {
  if (working) return;
  working = true;
  updateButtons();
  try {
    const data = await bridge.apiGet("connection");
    if (!data || typeof data.supported !== "boolean" || typeof data.testing !== "boolean") {
      throw new Error("Invalid connection metadata");
    }
    configuration = data;
    backendLabel.textContent = backendNames[data.backend] ?? "未知后端";
    modelLabel.textContent = typeof data.model === "string" && data.model ? data.model : "—";
    note.textContent = data.testing
      ? "已有测试正在执行，完成后请点击「刷新配置」。"
      : data.supported
        ? "将使用上方配置发送一次测试请求。修改配置后，请先保存并重载插件，再刷新本页。"
        : data.backend === "onnx"
          ? "当前为本地判断模型，无需测试远程连接。"
          : "判断后端配置无效，请检查配置并重载插件。";
    result.hidden = true;
  } catch {
    configuration = null;
    backendLabel.textContent = "读取失败";
    modelLabel.textContent = "—";
    note.textContent = "无法读取配置，请检查管理面板登录状态并刷新。";
    showResult("error", "配置读取失败", "请点击「刷新配置」重试；若仍失败，请重新打开插件页面。");
  } finally {
    working = false;
    updateButtons();
  }
}

testButton.addEventListener("click", async () => {
  if (working || !configuration?.supported || configuration.testing) return;
  working = true;
  updateButtons();
  testButton.textContent = "正在测试…";
  result.setAttribute("aria-busy", "true");
  showResult("pending", "正在测试", "正在验证模型连接和返回格式，请稍候。");
  try {
    const data = await bridge.apiPost("connection", {});
    if (!data || typeof data.success !== "boolean") throw new Error("Invalid test result");
    backendLabel.textContent = backendNames[data.backend] ?? "未知后端";
    modelLabel.textContent = typeof data.model === "string" && data.model ? data.model : "—";
    const duration = Number.isFinite(data.elapsed_ms) && data.elapsed_ms >= 0
      ? `耗时 ${Math.round(data.elapsed_ms)} 毫秒`
      : "";
    showResult(
      data.success ? "success" : "error",
      data.success ? "连接成功" : "连接测试失败",
      data.success
        ? "已成功调用模型，并取得有效的完整性判断结果。"
        : errors[data.code] ?? "测试未通过，请检查已保存的连接配置或稍后重试。",
      duration,
    );
  } catch {
    // Transport errors may contain upstream text; never echo raw exceptions.
    showResult("error", "请求未完成", "请检查管理面板登录状态；如有其他测试正在执行，请稍后刷新配置再试。");
  } finally {
    working = false;
    result.setAttribute("aria-busy", "false");
    testButton.textContent = "测试连接";
    updateButtons();
  }
});

refreshButton.addEventListener("click", refreshConfiguration);

try {
  if (!bridge) throw new Error("Plugin view bridge unavailable");
  await bridge.ready();
  await refreshConfiguration();
} catch {
  backendLabel.textContent = "管理面板未连接";
  note.textContent = "请从支持插件可视化视图的 AstrBot 管理面板打开本页面。";
  showResult("error", "页面初始化失败", "请刷新页面，或检查 AstrBot 与管理面板是否支持插件可视化视图。");
}
