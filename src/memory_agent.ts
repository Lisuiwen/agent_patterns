/**
 * 记忆智能体 (Memory Agent)
 * 
 * 功能概述：
 * 管理对话历史，当消息过多时自动压缩为摘要，避免上下文过长。
 * 实现长期记忆管理，平衡详细信息和计算成本。
 * 
 * 设计要点：
 * 1. 消息累积：使用 reducer 累积所有消息
 * 2. 自动压缩：当消息数量超过阈值时触发摘要生成
 * 3. 消息删除：压缩后删除旧消息，只保留最近的几条
 * 4. 摘要传递：将摘要传递给后续对话，保持上下文连续性
 * 5. 工作流模式：Start -> Chat -> [Summarize (条件) | End]
 * 
 * 适用场景：
 * - 长期对话系统（需要管理大量历史消息）
 * - 上下文窗口限制（需要压缩历史信息）
 * - 成本优化（减少 token 使用）
 * 
 * 扩展方向：
 * - 实现分层记忆（短期、长期、工作记忆）
 * - 基于重要性的选择性保留
 * - 外部记忆存储（数据库、向量库）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage, BaseMessage, RemoveMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：消息列表、摘要
const MemoryState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: (x, y) => x.concat(y), default: () => [] }), // 累积所有消息
  summary: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),            // 摘要（覆盖式更新）
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 }); // 适中的创造性

/**
 * 对话节点：基于历史消息和摘要生成回复
 * 设计要点：
 * - 如果有摘要，将其加入 SystemMessage，保持上下文连续性
 * - 使用所有历史消息作为上下文
 */
async function chatNode(state: typeof MemoryState.State) {
  const { messages, summary } = state;
  let systemPrompt = "你是一个健谈的 AI 朋友。";
  if (summary) systemPrompt += `\n这是你们之前的聊天摘要: "${summary}"`;
  const response = await model.invoke([new SystemMessage(systemPrompt), ...messages]);
  return { messages: [response] };
}

/**
 * 摘要节点：压缩历史消息为摘要，并删除旧消息
 * 设计要点：
 * - 合并旧摘要和新消息，生成新摘要
 * - 使用 RemoveMessage 删除旧消息（保留最近2条）
 * - 摘要保留关键信息，减少 token 使用
 */
async function summarizeNode(state: typeof MemoryState.State) {
  const { messages, summary } = state;
  console.log("\n🧹 [System] 历史消息过长，正在触发记忆压缩...");
  const summaryPrompt = `这是之前的对话摘要: "${summary}"\n这是新的几句对话:\n${messages.map(m => `${m.getType()}: ${m.content}`).join("\n")}\n请生成一个新的、合并后的简短摘要，涵盖所有关键信息。`;
  const response = await model.invoke([new HumanMessage(summaryPrompt)]);
  const newSummary = response.content as string;
  const deleteMessages = messages.slice(0, -2).map(m => new RemoveMessage({ id: m.id! })); // 删除除最后2条外的所有消息
  console.log(`✅ 新摘要: ${newSummary.slice(0, 30)}...`);
  return { summary: newSummary, messages: deleteMessages };
}

/**
 * 判断是否需要压缩：当消息数量超过阈值时触发摘要
 */
function shouldSummarize(state: typeof MemoryState.State) {
  return state.messages.length > 6 ? "summarize" : END;
}

/**
 * 构建工作流图
 * 关键设计：条件触发压缩
 * - 每次对话后检查消息数量
 * - 超过阈值则压缩，否则直接结束
 */
const workflow = new StateGraph(MemoryState)
  .addNode("chat", chatNode)                 // 对话节点
  .addNode("summarize", summarizeNode)        // 摘要节点
  .addEdge("__start__", "chat")             // 启动对话
  .addConditionalEdges("chat", shouldSummarize, {  // 条件判断
    summarize: "summarize",  // 需要压缩
    [END]: END               // 不需要压缩
  })
  .addEdge("summarize", END);                // 压缩完成后结束

const app = workflow.compile();

async function simulate() {
  const initialHistory = [
    new HumanMessage("我叫小明"), new BaseMessage({content: "你好小明", role: "assistant"}),
    new HumanMessage("喜欢足球"), new BaseMessage({content: "足球很棒", role: "assistant"}),
    new HumanMessage("住在北京"), new BaseMessage({content: "北京很大", role: "assistant"}),
    new HumanMessage("测试触发"), 
  ];
  console.log("🚀 模拟带记忆的对话...");
  const result = await app.invoke({ messages: initialHistory });
  if (result.summary) console.log(`🎉 成功触发记忆压缩！\n最终摘要: ${result.summary}`);
}
simulate().catch(console.error);
