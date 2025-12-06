import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const SafetyState = Annotation.Root({
  input: Annotation<string>,
  rawResponse: Annotation<string>,
  safetyStatus: Annotation<"SAFE" | "UNSAFE">,
  finalOutput: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

async function generateNode(state: typeof SafetyState.State) {
  const { input } = state;
  console.log(`\n🗣️ [Bot] 正在生成回复...`);
  const response = await model.invoke([new HumanMessage(input)]);
  return { rawResponse: response.content as string };
}

async function auditNode(state: typeof SafetyState.State) {
  const { rawResponse } = state;
  console.log(`\n👮 [Guard] 正在审计内容安全性...`);
  const prompt = `请审查以下内容是否包含敏感或违规信息。\n内容: "${rawResponse}"\n如果安全，请只回复 "SAFE"。如果不安全，请回复 "UNSAFE"。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  const status = res.content.toString().includes("UNSAFE") ? "UNSAFE" : "SAFE";
  console.log(`🛡️ 审计结果: ${status}`);
  return { safetyStatus: status as "SAFE" | "UNSAFE" };
}

async function sanitizeNode(state: typeof SafetyState.State) {
  const { rawResponse } = state;
  console.log(`\n🧼 [Sanitizer] 发现违规，正在重写...`);
  const prompt = `以下内容未能通过安全审查："${rawResponse}"\n请重写这段话，移除敏感信息。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { finalOutput: res.content as string };
}

async function passNode(state: typeof SafetyState.State) {
  return { finalOutput: state.rawResponse };
}

function routeLogic(state: typeof SafetyState.State) {
  return state.safetyStatus === "UNSAFE" ? "sanitize" : "pass";
}

const workflow = new StateGraph(SafetyState)
  .addNode("generate", generateNode)
  .addNode("audit", auditNode)
  .addNode("sanitize", sanitizeNode)
  .addNode("pass", passNode)
  .addEdge("__start__", "generate")
  .addEdge("generate", "audit")
  .addConditionalEdges("audit", routeLogic, { sanitize: "sanitize", pass: "pass" })
  .addEdge("sanitize", END)
  .addEdge("pass", END);

const app = workflow.compile();

async function main() {
  const input = "请帮我编一个故事，里面包含主角的电话号码是 13800138000，并且他在大骂邻居。";
  const result = await app.invoke({ input });
  console.log("\n====== 最终输出 ======\n" + result.finalOutput);
}
main().catch(console.error);

