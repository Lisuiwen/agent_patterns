/**
 * 思维树智能体 (Tree of Thoughts Agent)
 * 
 * 功能概述：
 * 实现 Tree of Thoughts (ToT) 算法：生成多个解题思路，评估每个思路的可行性，
 * 选择最佳思路并基于其生成最终解决方案。
 * 
 * 设计要点：
 * 1. 思路生成：生成多个不同的解题思路（发散思维）
 * 2. 思路评估：对每个思路进行评分和可行性分析
 * 3. 最优选择：选择得分最高的思路
 * 4. 方案生成：基于最佳思路生成完整解决方案
 * 5. 工作流模式：Start -> Propose -> Evaluate -> Solve -> End
 * 
 * 适用场景：
 * - 复杂问题求解（需要探索多种方案）
 * - 创新性任务（需要发散思维）
 * - 决策支持（需要评估多个选项）
 * 
 * 扩展方向：
 * - 实现多轮迭代（基于评估结果改进思路）
 * - 添加思路的详细展开（树状结构）
 * - 支持并行评估多个思路
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：问题、思路列表、评估列表、最佳思路、最终方案
const ToTState = Annotation.Root({
  problem: Annotation<string>,                                                      // 待解决的问题
  thoughts: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),  // 生成的思路列表
  evaluations: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }), // 每个思路的评估
  bestThought: Annotation<string>,                                                 // 最佳思路
  finalSolution: Annotation<string>,                                              // 最终解决方案
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 }); // 适中的创造性，鼓励思路多样性

/**
 * 思路生成节点：生成多个不同的解题思路
 * 设计要点：
 * - 要求思路"截然不同"，鼓励发散思维
 * - 使用 JSON 格式返回，便于解析
 */
async function proposeNode(state: typeof ToTState.State) {
  const { problem } = state;
  console.log(`\n🌱 [Proposer] 正在发散 3 种解题思路...`);
  const prompt = `用户问题: "${problem}"\n请提出 3 种截然不同的解决思路。请用 JSON 数组格式返回。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const content = response.content.toString().replace(/```json|```/g, "").trim();
  const thoughts = JSON.parse(content);
  return { thoughts };
}

/**
 * 评估节点：评估每个思路的可行性并打分
 * 设计要点：
 * - 顺序评估每个思路（实际应用可并行）
 * - 使用正则表达式提取分数
 * - 选择得分最高的思路作为最佳思路
 */
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

/**
 * 求解节点：基于最佳思路生成完整解决方案
 * 设计要点：使用选定的最佳思路作为指导，生成详细方案
 */
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
