/**
 * RAG 智能体 (Retrieval-Augmented Generation Agent)
 * 
 * 功能概述：
 * 实现检索增强生成模式：先从知识库检索相关信息，然后基于检索到的上下文生成回答。
 * 确保回答基于事实，而非仅依赖 LLM 的记忆。
 * 
 * 设计要点：
 * 1. 检索优先：先检索，后生成，确保信息准确性
 * 2. 上下文注入：将检索结果作为 prompt 的一部分，增强回答质量
 * 3. 知识库分离：知识库与 LLM 分离，便于更新和维护
 * 4. 工作流模式：Start -> Retrieve -> Generate -> End
 * 
 * 适用场景：
 * - 企业知识库问答（基于内部文档回答）
 * - 专业领域助手（需要准确的事实信息）
 * - 实时信息查询（结合外部数据源）
 * 
 * 扩展方向：
 * - 使用向量数据库（如 Pinecone、Weaviate）进行语义检索
 * - 实现多轮对话的上下文管理
 * - 添加引用来源功能
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：问题、检索到的上下文、最终答案
const RagState = Annotation.Root({
  question: Annotation<string>,                                                      // 用户问题
  context: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),     // 检索到的上下文
  answer: Annotation<string>,                                                       // 生成的答案
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 }); // temperature=0 确保基于事实的准确回答

// 模拟知识库（实际应用中应使用向量数据库或文档检索系统）
const MOCK_KNOWLEDGE_BASE = {
  "langgraph": "LangGraph 是一个用于构建有状态、多智能体应用程序的库，由 LangChain 开发。",
  "agent": "Agent 是一个使用 LLM 决定行动序列的系统。",
  "mcp": "Model Context Protocol (MCP) 是一个用于连接 AI 助手和系统的标准协议。"
};

/**
 * 检索节点：从知识库中检索与问题相关的上下文
 * 设计要点：
 * - 使用关键词匹配（实际应用应使用语义相似度搜索）
 * - 如果未找到相关信息，返回默认提示
 */
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

/**
 * 生成节点：基于检索到的上下文生成回答
 * 设计要点：
 * - 将检索到的 context 作为 prompt 的一部分
 * - LLM 基于提供的上下文回答，而非仅依赖训练数据
 * - 如果 context 为空，LLM 会明确告知无法基于知识库回答
 */
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
