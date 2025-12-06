/**
 * 安全护栏智能体 (Guardrails Agent) / 内容安全智能体
 * 
 * 功能概述：
 * 在输出前进行安全检查，如果内容不安全则进行清理或重写。
 * 实现内容安全控制，确保输出符合安全标准。
 * 
 * 设计要点：
 * 1. 生成-审核模式：先生成内容，再审核安全性
 * 2. 条件路由：根据审核结果决定是否清理
 * 3. 自动修复：发现不安全内容时自动重写
 * 4. 工作流模式：Start -> Generate -> Audit -> [Sanitize | Pass] -> End
 * 
 * 适用场景：
 * - 内容审核系统（防止有害内容输出）
 * - 合规性检查（确保符合法律法规）
 * - 敏感信息过滤（移除个人信息等）
 * 
 * 扩展方向：
 * - 使用专业的安全检测模型
 * - 实现多级安全检查
 * - 添加人工审核选项
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：输入、原始响应、安全状态、最终输出
const SafetyState = Annotation.Root({
  input: Annotation<string>,                    // 用户输入
  rawResponse: Annotation<string>,              // 原始生成的响应
  safetyStatus: Annotation<"SAFE" | "UNSAFE">,  // 安全审核状态
  finalOutput: Annotation<string>,             // 最终输出（清理后或原始）
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 }); // 适中的创造性

/**
 * 生成节点：生成原始响应
 * 设计要点：不进行任何过滤，生成完整响应
 */
async function generateNode(state: typeof SafetyState.State) {
  const { input } = state;
  console.log(`\n🗣️ [Bot] 正在生成回复...`);
  const response = await model.invoke([new HumanMessage(input)]);
  return { rawResponse: response.content as string };
}

/**
 * 审核节点：检查内容安全性
 * 设计要点：
 * - 使用 LLM 进行内容审核（实际应用应使用专业安全模型）
 * - 返回标准化的安全状态
 */
async function auditNode(state: typeof SafetyState.State) {
  const { rawResponse } = state;
  console.log(`\n👮 [Guard] 正在审计内容安全性...`);
  const prompt = `请审查以下内容是否包含敏感或违规信息。\n内容: "${rawResponse}"\n如果安全，请只回复 "SAFE"。如果不安全，请回复 "UNSAFE"。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  const status = res.content.toString().includes("UNSAFE") ? "UNSAFE" : "SAFE";
  console.log(`🛡️ 审计结果: ${status}`);
  return { safetyStatus: status as "SAFE" | "UNSAFE" };
}

/**
 * 清理节点：重写不安全的内容
 * 设计要点：移除敏感信息，保持内容完整性
 */
async function sanitizeNode(state: typeof SafetyState.State) {
  const { rawResponse } = state;
  console.log(`\n🧼 [Sanitizer] 发现违规，正在重写...`);
  const prompt = `以下内容未能通过安全审查："${rawResponse}"\n请重写这段话，移除敏感信息。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { finalOutput: res.content as string };
}

/**
 * 通过节点：安全内容直接通过
 */
async function passNode(state: typeof SafetyState.State) {
  return { finalOutput: state.rawResponse };
}

/**
 * 路由逻辑：根据安全状态决定是否清理
 */
function routeLogic(state: typeof SafetyState.State) {
  return state.safetyStatus === "UNSAFE" ? "sanitize" : "pass";
}

const workflow = new StateGraph(SafetyState)
  .addNode("generate", generateNode)
  .addNode("audit", auditNode)
  .addNode("sanitize", sanitizeNode)
  .addNode("pass", passNode)
  .addEdge("__start__", "generate")
  .addEdge("generate", "audit")
  .addConditionalEdges("audit", routeLogic, { sanitize: "sanitize", pass: "pass" })
  .addEdge("sanitize", END)
  .addEdge("pass", END);

const app = workflow.compile();

async function main() {
  const input = "请帮我编一个故事，里面包含主角的电话号码是 13800138000，并且他在大骂邻居。";
  const result = await app.invoke({ input });
  console.log("\n====== 最终输出 ======\n" + result.finalOutput);
}
main().catch(console.error);
