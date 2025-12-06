/**
 * 路由智能体 (Routing Agent)
 * 
 * 功能概述：
 * 根据用户请求的内容类型，智能路由到不同的专业处理节点。
 * 实现"一次路由，精准处理"的架构模式。
 * 
 * 设计要点：
 * 1. 智能分类：使用 LLM 分析用户意图，而非硬编码规则
 * 2. 条件路由：使用 addConditionalEdges 实现动态路由决策
 * 3. 专业化处理：每个处理节点都有专门的 SystemMessage 角色设定
 * 4. 工作流模式：Start -> Router -> [Tech/Life/General] -> End
 * 
 * 适用场景：
 * - 多领域客服系统（技术、生活、通用）
 * - 智能助手（根据问题类型选择专家）
 * - 内容分发系统（按类型路由到不同处理流程）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：请求内容、路由目标、最终响应
const RoutingState = Annotation.Root({
  request: Annotation<string>,      // 用户原始请求
  destination: Annotation<string>,  // 路由决策结果（tech_agent/life_agent/general_agent）
  response: Annotation<string>,     // 最终响应
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 }); // temperature=0 确保路由决策的确定性

/**
 * 路由节点：分析用户请求，决定路由到哪个专业处理节点
 * 设计要点：
 * - 使用 LLM 进行意图识别，比关键词匹配更智能
 * - 返回标准化的分类标签，便于后续路由
 */
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

/**
 * 技术专家节点：处理编程、技术相关的问题
 * 设计要点：通过 SystemMessage 设定专业角色，确保回答的专业性
 */
async function techNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`💻 [Tech Expert] 正在处理技术问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名资深架构师和代码专家。请用代码块和技术术语回答。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

/**
 * 生活顾问节点：处理生活、情感相关的问题
 * 设计要点：使用不同的语气和风格，体现专业化分工
 */
async function lifeNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`🌻 [Life Coach] 正在处理生活问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名温柔的生活顾问和心理学家。请用温暖、富有同理心的语气回答。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

/**
 * 通用助手节点：处理其他类型的问题
 */
async function generalNode(state: typeof RoutingState.State) {
  const { request } = state;
  console.log(`🌐 [General Bot] 正在处理通用问题...`);
  const response = await model.invoke([
    new SystemMessage("你是一名乐于助人的通用助手。"),
    new HumanMessage(request)
  ]);
  return { response: response.content as string };
}

/**
 * 路由逻辑函数：根据 routerNode 设置的 destination 决定下一步
 * 设计要点：这是条件边的核心，返回值必须匹配 addConditionalEdges 的映射键
 */
function routeLogic(state: typeof RoutingState.State) {
  return state.destination;
}

/**
 * 构建工作流图
 * 关键设计：使用 addConditionalEdges 实现动态路由
 * - router 节点完成后，根据 destination 值选择不同的处理节点
 */
const workflow = new StateGraph(RoutingState)
  .addNode("router", routerNode)              // 路由决策节点
  .addNode("tech_agent", techNode)            // 技术专家节点
  .addNode("life_agent", lifeNode)            // 生活顾问节点
  .addNode("general_agent", generalNode)       // 通用助手节点
  .addEdge("__start__", "router")             // 启动路由
  .addConditionalEdges("router", routeLogic, {  // 条件路由：根据 destination 选择
    tech_agent: "tech_agent",
    life_agent: "life_agent",
    general_agent: "general_agent"
  })
  .addEdge("tech_agent", END)                  // 各处理节点完成后结束
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
