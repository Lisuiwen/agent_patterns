import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage, BaseMessage, RemoveMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const MemoryState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: (x, y) => x.concat(y), default: () => [] }),
  summary: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }),
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 });

async function chatNode(state: typeof MemoryState.State) {
  const { messages, summary } = state;
  let systemPrompt = "你是一个健谈的 AI 朋友。";
  if (summary) systemPrompt += `\n这是你们之前的聊天摘要: "${summary}"`;
  const response = await model.invoke([new SystemMessage(systemPrompt), ...messages]);
  return { messages: [response] };
}

async function summarizeNode(state: typeof MemoryState.State) {
  const { messages, summary } = state;
  console.log("\n🧹 [System] 历史消息过长，正在触发记忆压缩...");
  const summaryPrompt = `这是之前的对话摘要: "${summary}"\n这是新的几句对话:\n${messages.map(m => `${m.getType()}: ${m.content}`).join("\n")}\n请生成一个新的、合并后的简短摘要，涵盖所有关键信息。`;
  const response = await model.invoke([new HumanMessage(summaryPrompt)]);
  const newSummary = response.content as string;
  const deleteMessages = messages.slice(0, -2).map(m => new RemoveMessage({ id: m.id! }));
  console.log(`✅ 新摘要: ${newSummary.slice(0, 30)}...`);
  return { summary: newSummary, messages: deleteMessages };
}

function shouldSummarize(state: typeof MemoryState.State) {
  return state.messages.length > 6 ? "summarize" : END;
}

const workflow = new StateGraph(MemoryState)
  .addNode("chat", chatNode)
  .addNode("summarize", summarizeNode)
  .addEdge("__start__", "chat")
  .addConditionalEdges("chat", shouldSummarize, { summarize: "summarize", [END]: END })
  .addEdge("summarize", END);

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
