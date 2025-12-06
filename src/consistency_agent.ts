import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const ConsistencyState = Annotation.Root({
  question: Annotation<string>,
  samples: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),
  finalAnswer: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 1.0 });

async function sampleNode(state: typeof ConsistencyState.State) {
  const { question } = state;
  const N = 3;
  console.log(`\n🎲 [Sampler] 正在进行 ${N} 次独立推理...`);
  const promises = Array(N).fill(0).map((_, i) => model.invoke([new HumanMessage(`问题: ${question}\n请一步步思考，最后并在末尾单独一行输出: "ANSWER: <你的最终答案>"`)]));
  const results = await Promise.all(promises);
  return { samples: results.map(r => r.content as string) };
}

async function voteNode(state: typeof ConsistencyState.State) {
  const { samples, question } = state;
  console.log(`\n🗳️ [Voter] 正在统计票数...`);
  const prompt = `这里有针对问题 "${question}" 的 3 个不同解答：\n${samples.map((s, i) => `--- 解答 ${i+1} ---\n${s}\n`).join("\n")}\n请分析这些解答。虽然过程可能不同，但结论是否一致？请输出最正确、最一致的那个结论。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { finalAnswer: response.content as string };
}

const workflow = new StateGraph(ConsistencyState)
  .addNode("sample", sampleNode)
  .addNode("vote", voteNode)
  .addEdge("__start__", "sample")
  .addEdge("sample", "vote")
  .addEdge("vote", END);

const app = workflow.compile();

async function main() {
  const question = "农场里有鸡和兔子共 35 个头，94 只脚。请问鸡和兔子各多少只？";
  const result = await app.invoke({ question });
  console.log("\n====== 多数投票结果 ======\n" + result.finalAnswer);
}
main().catch(console.error);
