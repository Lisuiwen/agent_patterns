import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage, BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const TeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: (x, y) => x.concat(y), default: () => [] }),
  next: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "Supervisor" }),
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 });

async function researcherNode(state: typeof TeamState.State) {
  console.log("🕵️ [Researcher] 正在搜集信息...");
  const lastMessage = state.messages[state.messages.length - 1];
  const response = await model.invoke([new SystemMessage("你是一个研究员。请提供关于用户问题的准确数据。"), lastMessage]);
  return { messages: [response] };
}

async function writerNode(state: typeof TeamState.State) {
  console.log("✍️ [Writer] 正在撰写文案...");
  const lastMessage = state.messages[state.messages.length - 1];
  const response = await model.invoke([new SystemMessage("你是一个作家。请基于之前的研究结果，写一段优美的文字。"), lastMessage]);
  return { messages: [response] };
}

async function supervisorNode(state: typeof TeamState.State) {
  console.log("👮 [Supervisor] 正在调度...");
  const { messages } = state;
  const systemPrompt = `你是一个团队管理者。团队成员有: "Researcher", "Writer"。\n规则:\n1. 如果用户的问题需要事实支撑，先让 "Researcher" 工作。\n2. 有了资料后，让 "Writer" 进行写作。\n3. 如果写作已完成且质量尚可，回复 "FINISH"。\n只返回一个单词: "Researcher", "Writer", 或 "FINISH"。`;
  const response = await model.invoke([new SystemMessage(systemPrompt), ...messages]);
  const decision = response.content.toString().trim().replace(/['"]/g, '');
  console.log(`👮 决策: ${decision}`);
  return { next: decision };
}

function routeLogic(state: typeof TeamState.State) {
  if (state.next === "Researcher") return "researcher";
  if (state.next === "Writer") return "writer";
  return END;
}

const workflow = new StateGraph(TeamState)
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherNode)
  .addNode("writer", writerNode)
  .addEdge("__start__", "supervisor")
  .addConditionalEdges("supervisor", routeLogic, { researcher: "researcher", writer: "writer", [END]: END })
  .addEdge("researcher", "supervisor")
  .addEdge("writer", "supervisor");

const app = workflow.compile();

async function main() {
  const task = "请帮我写一段关于'量子计算'的简短介绍，风格要科幻一点。";
  console.log(`🚀 开始团队协作任务: ${task}`);
  const result = await app.invoke({ messages: [new HumanMessage(task)] });
  const lastMsg = result.messages[result.messages.length - 1];
  console.log("\n====== 最终产出 ======\n" + lastMsg.content);
}
main().catch(console.error);

