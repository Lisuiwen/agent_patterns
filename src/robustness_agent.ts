/**
 * 健壮性智能体 (Robustness Agent) / 容错智能体
 * 
 * 功能概述：
 * 实现重试机制和降级策略，当主处理节点失败时自动重试，多次失败后启用备用方案。
 * 提高系统的可靠性和容错能力。
 * 
 * 设计要点：
 * 1. 重试机制：主节点失败后自动重试（最多3次）
 * 2. 错误累积：记录所有失败尝试的错误信息
 * 3. 降级策略：多次失败后切换到备用节点
 * 4. 条件循环：根据结果和尝试次数决定下一步
 * 5. 工作流模式：Start -> Primary (循环重试) -> [Primary | Fallback] -> End
 * 
 * 适用场景：
 * - 不稳定服务调用（网络 API、外部服务）
 * - 高可靠性要求（不能因单次失败而中断）
 * - 容错系统（需要优雅降级）
 * 
 * 扩展方向：
 * - 实现指数退避重试
 * - 添加多个备用节点（多级降级）
 * - 支持错误类型分析和针对性处理
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：任务、尝试次数、错误列表、结果
const RobustState = Annotation.Root({
  task: Annotation<string>,                                                      // 用户任务
  attempts: Annotation<number>({ reducer: (x, y) => y, default: () => 0 }),    // 尝试次数（覆盖式更新）
  errors: Annotation<string[]>({ reducer: (x, y) => x.concat(y), default: () => [] }), // 错误列表（累积）
  result: Annotation<string>,                                                   // 最终结果
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 }); // 适中的创造性

/**
 * 主处理节点：执行任务（可能失败）
 * 设计要点：
 * - 模拟不稳定的服务（80% 失败率，前2次）
 * - 失败时记录错误并增加尝试次数
 * - 成功时返回结果
 */
async function unstableToolNode(state: typeof RobustState.State) {
  const { attempts, task } = state;
  console.log(`\n⚡ [Primary Tool] 尝试第 ${attempts + 1} 次执行: "${task}"`);
  const isFailure = Math.random() > 0.2;
  if (isFailure && attempts < 2) {
    console.error("   ❌ 调用失败：网络超时或服务不可用。");
    return { attempts: attempts + 1, errors: [`Attempt ${attempts + 1}: Connection Timeout`] };
  }
  console.log("   ✅ 调用成功！");
  const response = await model.invoke([new SystemMessage("你是一个主处理单元。请处理用户任务。"), new HumanMessage(task)]);
  return { result: response.content as string, attempts: attempts + 1 };
}

/**
 * 备用节点：主节点多次失败后的降级方案
 * 设计要点：
 * - 使用简化的处理逻辑
 * - 明确标识为备用模式响应
 * - 基于历史错误信息进行优化
 */
async function fallbackNode(state: typeof RobustState.State) {
  const { task, errors } = state;
  console.log(`\n🛡️ [Fallback] 主节点多次失败，启用备用方案...\n   历史错误: ${errors.join(", ")}`);
  const prompt = `主系统已崩溃。你是一个备用系统 (Safe Mode)。请用最简短、最安全的方式回应用户任务: "${task}"\n并在开头注明 "[备用模式响应]"`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { result: response.content as string };
}

/**
 * 路由逻辑：决定重试、降级还是完成
 * 设计要点：
 * - 如果已有结果，直接结束
 * - 如果尝试次数 >= 3，启用备用方案
 * - 否则继续重试主节点
 */
function routeLogic(state: typeof RobustState.State) {
  if (state.result) return END;
  if (state.attempts >= 3) return "fallback";
  return "primary_tool";
}

const workflow = new StateGraph(RobustState)
  .addNode("primary_tool", unstableToolNode)
  .addNode("fallback", fallbackNode)
  .addEdge("__start__", "primary_tool")
  .addConditionalEdges("primary_tool", routeLogic, { primary_tool: "primary_tool", fallback: "fallback", [END]: END })
  .addEdge("fallback", END);

const app = workflow.compile();

async function main() {
  console.log("🚀 开始任务：模拟不稳定环境...");
  const finalState = await app.invoke({ task: "分析 2024 年 Q3 财报数据" });
  console.log("\n====== 最终结果 ======\n" + finalState.result);
}
main().catch(console.error);
