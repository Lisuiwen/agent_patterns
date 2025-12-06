/**
 * 探索智能体 (Exploration Agent) / 假设验证智能体
 * 
 * 功能概述：
 * 模拟科学研究流程：提出假设 -> 验证假设 -> 生成报告。
 * 用于探索性任务，生成创新想法并验证其可行性。
 * 
 * 设计要点：
 * 1. 假设生成：使用 LLM 生成创新性假设
 * 2. 实验验证：对每个假设进行验证（模拟实验）
 * 3. 结果累积：收集所有验证结果
 * 4. 报告生成：基于所有发现生成综合报告
 * 5. 工作流模式：Start -> Hypothesis -> Experiment -> Report -> End
 * 
 * 适用场景：
 * - 研究探索（提出新研究方向）
 * - 产品创新（生成并验证新想法）
 * - 问题诊断（提出可能原因并验证）
 * 
 * 扩展方向：
 * - 实现迭代假设生成（基于验证结果改进假设）
 * - 添加实验设计节点
 * - 支持外部数据源验证
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：领域、假设列表、发现列表、最终报告
const ExplorationState = Annotation.Root({
  domain: Annotation<string>,                                                      // 探索领域
  hypotheses: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }), // 生成的假设
  findings: Annotation<string[]>({ reducer: (x, y) => x.concat(y), default: () => [] }), // 验证发现（累积）
  finalReport: Annotation<string>,                                                 // 最终报告
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.8 }); // 高 temperature 鼓励创新性假设

/**
 * 假设生成节点：针对领域提出创新性假设
 * 设计要点：
 * - 使用 JSON 格式返回，便于解析
 * - 要求假设具有创新性和探索性
 */
async function hypothesisNode(state: typeof ExplorationState.State) {
  const { domain } = state;
  console.log(`\n💡 [Explorer] 正在对 "${domain}" 领域提出假设...`);
  const prompt = `你是一个前沿研究员。针对领域 "${domain}"，请提出 2 个具有创新性、大胆的假设或研究方向。\n格式：JSON字符串数组，如 ["假设A...", "假设B..."]`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  const text = res.content.toString().replace(/```json|```/g, "").trim();
  const hypotheses = JSON.parse(text);
  console.log(`   生成的假设: \n   1. ${hypotheses[0]}\n   2. ${hypotheses[1]}`);
  return { hypotheses };
}

/**
 * 实验节点：验证每个假设
 * 设计要点：
 * - 顺序验证每个假设（实际应用可并行）
 * - 基于 LLM 的知识库进行验证（模拟实验）
 * - 累积所有发现到 findings 数组
 */
async function experimentNode(state: typeof ExplorationState.State) {
  const { hypotheses } = state;
  console.log(`\n🔬 [Scientist] 正在验证假设...`);
  const newFindings = [];
  for (const hyp of hypotheses) {
    const prompt = `假设: "${hyp}"\n请模拟对这个假设进行验证。基于你现有的知识库，判断这个假设成立的可能性，并给出一个结论。`;
    const res = await model.invoke([new HumanMessage(prompt)]);
    console.log(`   🧪 验证完成: ${hyp.slice(0, 15)}...`);
    newFindings.push(`针对假设 [${hyp}] 的发现: ${res.content}`);
  }
  return { findings: newFindings };
}

/**
 * 报告节点：基于所有发现生成综合报告
 */
async function reportNode(state: typeof ExplorationState.State) {
  const { domain, findings } = state;
  console.log(`\n📝 [Reporter] 正在撰写发现报告...`);
  const prompt = `领域: ${domain}\n基于以下实验发现:\n${findings.join("\n\n")}\n请写一份简短的《前沿探索报告》，总结我们发现的新知。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { finalReport: res.content as string };
}

const workflow = new StateGraph(ExplorationState)
  .addNode("hypothesis_gen", hypothesisNode)
  .addNode("experiment", experimentNode)
  .addNode("report_gen", reportNode)
  .addEdge("__start__", "hypothesis_gen")
  .addEdge("hypothesis_gen", "experiment")
  .addEdge("experiment", "report_gen")
  .addEdge("report_gen", END);

const app = workflow.compile();

async function main() {
  const result = await app.invoke({ domain: "火星上的微生物生命存在形式" });
  console.log("\n====== 探索报告 ======\n" + result.finalReport);
}
main().catch(console.error);
