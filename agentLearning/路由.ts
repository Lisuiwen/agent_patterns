import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const RoutingState = Annotation.Root({
  request: Annotation<string>,
  destination: Annotation<string>,
  response: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 });

async function routerNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`\n🧭 [Router] 正在分析用户意图: "${request}"`);
  const prompt = `你是一个路由助手。请分析用户的请求，将其归类为以下之一：
  - "TECH": 如果是关于编程、代码、计算机技术的问题。
  - "LIFE": 如果是关于生活建议、情感、烹饪等问题。
  - "GENERAL": 其他所有问题。
  只返回分类关键词，不要包含其他字符。`;
  const response = await model.invoke([new SystemMessage(prompt), new HumanMessage(request)]);
  const category = response.content.toString().trim().toUpperCase();
  let destination = "general_agent";
  if (category.includes("TECH")) destination = "tech_agent";
  else if (category.includes("LIFE")) destination = "life_agent";
  console.log(`🔀 [Router] 分流至: ${destination}`);
  return { destination };
}

async function techNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`💻 [Tech Expert] 正在处理技术问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名资深架构师和代码专家。请用代码块和技术术语回答。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

async function lifeNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`🌻 [Life Coach] 正在处理生活问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名温柔的生活顾问和心理学家。请用温暖、富有同理心的语气回答。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

async function generalNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`🌐 [General Bot] 正在处理通用问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名乐于助人的通用助手。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

function routeLogic(state: typeof RoutingState.State) {
  return state.destination;
}

const workflow = new StateGraph(RoutingState)
  .addNode("router", routerNode)
  .addNode("tech_agent", techNode)
  .addNode("life_agent", lifeNode)
  .addNode("general_agent", generalNode)
  .addEdge("__start__", "router")
  .addConditionalEdges("router", routeLogic, {
    tech_agent: "tech_agent",
    life_agent: "life_agent",
    general_agent: "general_agent"
  })
  .addEdge("tech_agent", END)
  .addEdge("life_agent", END)
  .addEdge("general_agent", END);

const app = workflow.compile();

async function main() {
  const inputs = ["如何用 Python 实现快速排序？", "最近心情很焦虑，怎么缓解压力？", "天空为什么是蓝色的？"];
  for (const input of inputs) {
    console.log(`\n--- New Request: ${input} ---`);
    const finalState = await app.invoke({ request: input });
    console.log(`✅ [Response]: ${finalState.response.slice(0, 50)}...`);
  }
}
main().catch(console.error);

