import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const HitlState = Annotation.Root({
  task: Annotation<string>,
  draft: Annotation<string>,
  feedback: Annotation<string>,
  finalResult: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

async function writeNode(state: typeof HitlState.State) {
  const { task, feedback, draft } = state;
  if (feedback) {
    console.log(`\n✍️ [Writer] 根据人类反馈修改中: "${feedback}"`);
    const prompt = `之前的草稿: ${draft}\n人类反馈: ${feedback}\n请根据反馈修改草稿。`;
    const res = await model.invoke([new HumanMessage(prompt)]);
    return { draft: res.content as string, feedback: "" };
  } else {
    console.log(`\n✍️ [Writer] 初次撰写: ${task}`);
    const res = await model.invoke([new HumanMessage(`请为任务写一篇简短的邮件草稿: ${task}`)]);
    return { draft: res.content as string };
  }
}

async function mockHumanNode(state: typeof HitlState.State) {
  console.log("\n🛑 [Mock Human] 看到草稿: " + state.draft.slice(0, 20) + "...");
  if (!state.feedback) {
    console.log("👤 人类: 不太行，语气要更正式一点。");
    return { feedback: "语气要更正式一点" };
  } else {
    console.log("👤 人类: 这次可以了，approve。");
    return { feedback: "approve" };
  }
}

async function sendNode(state: typeof HitlState.State) {
  console.log("\n📤 [Sender] 邮件已发送！(模拟)");
  return { finalResult: "SENT" };
}

function router(state: typeof HitlState.State) {
  if (state.feedback === "approve") return "sender";
  if (state.feedback) return "writer";
  return "sender";
}

const hitlWorkflow = new StateGraph(HitlState)
  .addNode("writer", writeNode)
  .addNode("human", mockHumanNode)
  .addNode("sender", sendNode)
  .addEdge("__start__", "writer")
  .addEdge("writer", "human")
  .addConditionalEdges("human", router, { writer: "writer", sender: "sender" })
  .addEdge("sender", END);

const app = hitlWorkflow.compile();
async function main() { await app.invoke({ task: "向老板请假去滑雪" }); }
main().catch(console.error);
