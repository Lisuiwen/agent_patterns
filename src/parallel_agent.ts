/**
 * 并行智能体 (Parallel Agent)
 * 
 * 功能概述：
 * 通过并行执行多个独立的智能体任务，然后聚合结果，实现多角度分析和决策。
 * 适用于需要同时从不同视角分析问题的场景，如辩论、利弊分析、多方案对比等。
 * 
 * 设计要点：
 * 1. 并行执行：positive 和 negative 节点同时运行，提高效率
 * 2. 状态聚合：使用 aggregator 节点合并并行结果
 * 3. 角色分工：不同节点扮演不同角色（乐观主义者 vs 批判性思维者）
 * 4. 工作流模式：Start -> [Positive, Negative] (并行) -> Aggregator -> End
 * 
 * 适用场景：
 * - 决策支持系统（需要多角度分析）
 * - 内容审核（同时检查优点和风险）
 * - 产品评估（功能优势 vs 潜在问题）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态结构：包含主题、支持观点、反对观点和最终总结
const ParallelState = Annotation.Root({
  topic: Annotation<string>,                    // 待分析的主题
  pros: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),  // 支持观点（使用 reducer 确保只保留最新值）
  cons: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }), // 反对观点
  finalSummary: Annotation<string>,            // 综合总结
});

// LLM 配置：使用 Moonshot API (Kimi 模型)
const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 }); // temperature=0.7 允许一定创造性

/**
 * 正面分析节点：从乐观角度分析主题的优点
 * 设计要点：使用 SystemMessage 设定角色，引导 LLM 从特定视角思考
 */
async function positiveNode(state: typeof ParallelState.State) {
  console.log("🟢 [Positive Agent] 正在生成支持观点...");
  const response = await model.invoke([
    new SystemMessage("你是一个乐观主义者。请列出该主题的3个主要优点。"),
    new HumanMessage(state.topic)
  ]);
  return { pros: response.content as string };
}

/**
 * 负面分析节点：从批判性角度分析主题的风险和缺点
 * 设计要点：与 positiveNode 并行执行，提供对立视角
 */
async function negativeNode(state: typeof ParallelState.State) {
  console.log("🔴 [Negative Agent] 正在生成反对观点...");
  const response = await model.invoke([
    new SystemMessage("你是一个批判性思维者。请列出该主题的3个潜在风险或缺点。"),
    new HumanMessage(state.topic)
  ]);
  return { cons: response.content as string };
}

/**
 * 聚合节点：综合正反两方观点，生成平衡的总结报告
 * 设计要点：等待并行节点完成后执行，整合所有信息
 */
async function aggregatorNode(state: typeof ParallelState.State) {
  console.log("🔗 [Aggregator] 正在合并报告...");
  const { topic, pros, cons } = state;
  const prompt = `用户询问主题: "${topic}"\n支持方观点:\n${pros}\n反对方观点:\n${cons}\n请综合以上两方观点，写一段平衡的总结报告。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { finalSummary: response.content as string };
}

/**
 * 构建工作流图
 * 关键设计：从 __start__ 同时连接到 positive 和 negative，实现真正的并行执行
 * LangGraph 会自动等待所有输入边完成后再执行 aggregator
 */
const workflow = new StateGraph(ParallelState)
  .addNode("positive", positiveNode)      // 正面分析节点
  .addNode("negative", negativeNode)       // 负面分析节点
  .addNode("aggregator", aggregatorNode)  // 聚合节点
  .addEdge("__start__", "positive")       // 启动时同时触发两个并行节点
  .addEdge("__start__", "negative")
  .addEdge("positive", "aggregator")      // 两个节点都完成后才能执行聚合
  .addEdge("negative", "aggregator")
  .addEdge("aggregator", END);            // 完成

const app = workflow.compile();

// 导出 app 供服务器使用
export { app };

// 只在直接运行时执行 main 函数
async function main() {
  const topic = "AI 是否会完全取代程序员";
  console.log(`🚀 开始并行辩论，主题: ${topic}`);
  const result = await app.invoke({ topic });
  console.log("\n====== 🟢 正方 ======"); console.log(result.pros);
  console.log("\n====== 🔴 反方 ======"); console.log(result.cons);
  console.log("\n====== 🔗 综合总结 ======"); console.log(result.finalSummary);
}

// 检查是否是直接运行该文件（而非被导入）
if (require.main === module) {
  main().catch(console.error);
}
