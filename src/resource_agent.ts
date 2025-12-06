import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const ResourceState = Annotation.Root({
  task: Annotation<string>,
  complexity: Annotation<"SIMPLE" | "COMPLEX">,
  cost: Annotation<number>,
  response: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const baseModel = new ChatOpenAI({ ...CONFIG, temperature: 0 });

async function classifierNode(state: typeof ResourceState.State) {
  const { task } = state;
  console.log(`\n⚖️ [Classifier] 正在评估任务复杂度: "${task}"`);
  const prompt = `请评估以下任务的复杂度。\n如果任务涉及简单的问候、翻译、事实查询，返回 "SIMPLE"。\n如果任务涉及逻辑推理、代码编写、创意写作，返回 "COMPLEX"。\n只返回一个单词。`;
  const res = await baseModel.invoke([new HumanMessage(prompt), new HumanMessage(task)]);
  const complexity = res.content.toString().includes("COMPLEX") ? "COMPLEX" : "SIMPLE";
  console.log(`   判定结果: ${complexity}`);
  return { complexity, cost: 0.1 };
}

async function cheapModelNode(state: typeof ResourceState.State) {
  console.log(`\n⚡ [Flash Model] 使用高速低成本模型处理...`);
  const res = await baseModel.invoke([new SystemMessage("你是一个追求速度的助手。请用最简练的话回答。"), new HumanMessage(state.task)]);
  return { response: res.content as string, cost: 0.5 };
}

async function expensiveModelNode(state: typeof ResourceState.State) {
  console.log(`\n🐢 [Pro Model] 使用深度推理模型处理...`);
  const res = await baseModel.invoke([new SystemMessage("你是一个深度思考的专家。请详细、全面地回答，展示你的推理能力。"), new HumanMessage(state.task)]);
  return { response: res.content as string, cost: 10.0 };
}

function routeLogic(state: typeof ResourceState.State) {
  return state.complexity === "COMPLEX" ? "expensive" : "cheap";
}

const workflow = new StateGraph(ResourceState)
  .addNode("classifier", classifierNode)
  .addNode("cheap", cheapModelNode)
  .addNode("expensive", expensiveModelNode)
  .addEdge("__start__", "classifier")
  .addConditionalEdges("classifier", routeLogic, { cheap: "cheap", expensive: "expensive" })
  .addEdge("cheap", END)
  .addEdge("expensive", END);

const app = workflow.compile();

async function main() {
  const tasks = ["你好，早上好！", "请设计一个基于微服务架构的电商系统，并给出数据库ER图描述"];
  let totalCost = 0;
  for (const task of tasks) {
    const res = await app.invoke({ task });
    console.log(`💬 回复: ${res.response.slice(0, 50)}...\n💰 本次花费: ${res.cost}`);
    totalCost += res.cost;
  }
  console.log(`\n============== 总花费: ${totalCost} ==============`);
}
main().catch(console.error);
