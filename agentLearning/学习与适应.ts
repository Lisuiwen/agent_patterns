import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const EXPERIENCE_DB: string[] = ["经验1: 用户喜欢简练的回答。", "经验2: 如果涉及代码，必须给出 TypeScript 类型定义。"];

const LearningState = Annotation.Root({
  task: Annotation<string>,
  retrievedContext: Annotation<string>,
  result: Annotation<string>,
  newInsight: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 });

async function recallNode(state: typeof LearningState.State) {
  console.log(`\n📖 [Memory] 正在检索过往经验...`);
  const context = EXPERIENCE_DB.join("\n");
  return { retrievedContext: context };
}

async function actNode(state: typeof LearningState.State) {
  const { task, retrievedContext } = state;
  console.log(`\n✍️ [Actor] 正在执行任务...`);
  const prompt = `你是一个智能助手。请执行用户任务。\n⚠️ 重要：请务必遵守以下过往经验教训：\n${retrievedContext}\n用户任务: "${task}"`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { result: res.content as string };
}

async function learnNode(state: typeof LearningState.State) {
  const { task, result } = state;
  console.log(`\n🧠 [Learner] 正在总结本次教训...`);
  const prompt = `任务: "${task}"\n回答: "${result}"\n请反思这次任务，提取一条通用的"最佳实践"或"注意事项"，简短一点。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  const insight = res.content as string;
  EXPERIENCE_DB.push(`新经验: ${insight}`);
  console.log(`✅ 已通过学习获得新知识: "${insight}"`);
  return { newInsight: insight };
}

const workflow = new StateGraph(LearningState)
  .addNode("recall", recallNode)
  .addNode("act", actNode)
  .addNode("learn", learnNode)
  .addEdge("__start__", "recall")
  .addEdge("recall", "act")
  .addEdge("act", "learn")
  .addEdge("learn", END);

const app = workflow.compile();

async function main() {
  await app.invoke({ task: "请用 JS 写一个求和函数" });
  await app.invoke({ task: "请写一个打招呼的函数" });
  console.log("\n📚 当前经验库状态:", EXPERIENCE_DB);
}
main().catch(console.error);

