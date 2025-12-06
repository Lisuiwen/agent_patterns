import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const ParallelState = Annotation.Root({
  topic: Annotation<string>,
  pros: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),
  cons: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),
  finalSummary: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

async function positiveNode(state: typeof ParallelState.State) {
  console.log("🟢 [Positive Agent] 正在生成支持观点...");
  const response = await model.invoke([
    new SystemMessage("你是一个乐观主义者。请列出该主题的3个主要优点。"),
    new HumanMessage(state.topic)
  ]);
  return { pros: response.content as string };
}

async function negativeNode(state: typeof ParallelState.State) {
  console.log("🔴 [Negative Agent] 正在生成反对观点...");
  const response = await model.invoke([
    new SystemMessage("你是一个批判性思维者。请列出该主题的3个潜在风险或缺点。"),
    new HumanMessage(state.topic)
  ]);
  return { cons: response.content as string };
}

async function aggregatorNode(state: typeof ParallelState.State) {
  console.log("🔗 [Aggregator] 正在合并报告...");
  const { topic, pros, cons } = state;
  const prompt = `用户询问主题: "${topic}"\n支持方观点:\n${pros}\n反对方观点:\n${cons}\n请综合以上两方观点，写一段平衡的总结报告。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { finalSummary: response.content as string };
}

const workflow = new StateGraph(ParallelState)
  .addNode("positive", positiveNode)
  .addNode("negative", negativeNode)
  .addNode("aggregator", aggregatorNode)
  .addEdge("__start__", "positive")
  .addEdge("__start__", "negative")
  .addEdge("positive", "aggregator")
  .addEdge("negative", "aggregator")
  .addEdge("aggregator", END);

const app = workflow.compile();

async function main() {
  const topic = "AI 是否会完全取代程序员";
  console.log(`🚀 开始并行辩论，主题: ${topic}`);
  const result = await app.invoke({ topic });
  console.log("\n====== 🟢 正方 ======"); console.log(result.pros);
  console.log("\n====== 🔴 反方 ======"); console.log(result.cons);
  console.log("\n====== 🔗 综合总结 ======"); console.log(result.finalSummary);
}
main().catch(console.error);
