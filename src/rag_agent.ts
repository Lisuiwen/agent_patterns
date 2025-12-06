import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const RagState = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),
  answer: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 });

const MOCK_KNOWLEDGE_BASE = {
  "langgraph": "LangGraph 是一个用于构建有状态、多智能体应用程序的库，由 LangChain 开发。",
  "agent": "Agent 是一个使用 LLM 决定行动序列的系统。",
  "mcp": "Model Context Protocol (MCP) 是一个用于连接 AI 助手和系统的标准协议。"
};

async function retrieveNode(state: typeof RagState.State) {
  const { question } = state;
  console.log(`\n🔍 [Retriever] 正在检索知识库: "${question}"`);
  let context = "未找到相关信息。";
  const lowerQ = question.toLowerCase();
  if (lowerQ.includes("langgraph")) context = MOCK_KNOWLEDGE_BASE["langgraph"];
  else if (lowerQ.includes("agent")) context = MOCK_KNOWLEDGE_BASE["agent"];
  else if (lowerQ.includes("mcp")) context = MOCK_KNOWLEDGE_BASE["mcp"];
  console.log(`📄 检索结果: ${context}`);
  return { context };
}

async function generateNode(state: typeof RagState.State) {
  const { question, context } = state;
  console.log(`\n🧠 [Generator] 正在生成回答...`);
  const prompt = `请基于以下上下文回答用户问题。\n上下文:\n${context}\n用户问题: ${question}`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { answer: response.content as string };
}

const workflow = new StateGraph(RagState)
  .addNode("retrieve", retrieveNode)
  .addNode("generate", generateNode)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", END);

const app = workflow.compile();

async function main() {
  const questions = ["LangGraph 是什么？", "今天天气怎么样？"];
  for (const q of questions) {
    console.log(`\n--- Query: ${q} ---`);
    const res = await app.invoke({ question: q });
    console.log(`💬 回答: ${res.answer}`);
  }
}
main().catch(console.error);
