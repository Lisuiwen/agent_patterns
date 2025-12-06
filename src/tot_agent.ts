import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const ToTState = Annotation.Root({
  problem: Annotation<string>,
  thoughts: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),
  evaluations: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),
  bestThought: Annotation<string>,
  finalSolution: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

async function proposeNode(state: typeof ToTState.State) {
  const { problem } = state;
  console.log(`\n🌱 [Proposer] 正在发散 3 种解题思路...`);
  const prompt = `用户问题: "${problem}"\n请提出 3 种截然不同的解决思路。请用 JSON 数组格式返回。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const content = response.content.toString().replace(/```json|```/g, "").trim();
  const thoughts = JSON.parse(content);
  return { thoughts };
}

async function evaluateNode(state: typeof ToTState.State) {
  const { problem, thoughts } = state;
  console.log(`\n⚖️ [Evaluator] 正在评估每个思路的可行性...`);
  const evaluations = [];
  let bestThought = thoughts[0];
  let maxScore = -1;
  for (const thought of thoughts) {
    const prompt = `问题: ${problem}\n解决思路: ${thought}\n请评估这个思路的可行性。最后给出一个 0-10 的整数打分。格式: "分析内容... SCORE: 8"`;
    const res = await model.invoke([new HumanMessage(prompt)]);
    const content = res.content as string;
    evaluations.push(content);
    const match = content.match(/SCORE:\s*(\d+)/);
    const score = match ? parseInt(match[1]) : 0;
    console.log(`📊 思路得分: ${score}`);
    if (score > maxScore) { maxScore = score; bestThought = thought; }
  }
  console.log(`🏆 最佳思路 (Score ${maxScore}): ${bestThought.slice(0, 30)}...`);
  return { evaluations, bestThought };
}

async function solveNode(state: typeof ToTState.State) {
  const { problem, bestThought } = state;
  console.log(`\n🚀 [Solver] 正在基于最佳思路解题...`);
  const prompt = `问题: ${problem}\n选定的最佳思路: ${bestThought}\n请根据这个思路，写出完整的解决方案。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { finalSolution: res.content as string };
}

const workflow = new StateGraph(ToTState)
  .addNode("propose", proposeNode)
  .addNode("evaluate", evaluateNode)
  .addNode("solve", solveNode)
  .addEdge("__start__", "propose")
  .addEdge("propose", "evaluate")
  .addEdge("evaluate", "solve")
  .addEdge("solve", END);

const app = workflow.compile();

async function main() {
  const problem = "如何在一周内策划一场吸引 1000 人参与的线上技术讲座？预算只有 500 元。";
  const result = await app.invoke({ problem });
  console.log("\n====== 最终方案 ======\n" + result.finalSolution);
}
main().catch(console.error);
