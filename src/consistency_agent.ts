/**
 * 一致性智能体 (Consistency Agent) / 多数投票智能体
 * 
 * 功能概述：
 * 通过多次独立推理生成多个答案，然后通过投票机制选择最一致的答案。
 * 提高答案的可靠性和准确性，减少随机性带来的错误。
 * 
 * 设计要点：
 * 1. 多次采样：并行生成多个独立答案（使用高 temperature 增加多样性）
 * 2. 投票机制：使用 LLM 分析所有答案，选择最一致、最正确的结论
 * 3. 并行执行：所有采样同时进行，提高效率
 * 4. 工作流模式：Start -> Sample (并行N次) -> Vote -> End
 * 
 * 适用场景：
 * - 重要决策（需要高可靠性）
 * - 数学/逻辑问题（需要准确答案）
 * - 减少幻觉（通过一致性检查）
 * 
 * 扩展方向：
 * - 实现加权投票（根据答案质量加权）
 * - 添加置信度评分
 * - 支持不同模型的投票（模型集成）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：问题、多个样本答案、最终答案
const ConsistencyState = Annotation.Root({
  question: Annotation<string>,                                                      // 用户问题
  samples: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }), // 多个独立推理结果
  finalAnswer: Annotation<string>,                                                 // 投票后的最终答案
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 1.0 }); // 高 temperature 增加答案多样性

/**
 * 采样节点：并行生成多个独立答案
 * 设计要点：
 * - 使用 Promise.all 实现真正的并行执行
 * - 高 temperature 确保答案的多样性
 * - 要求答案格式统一（末尾包含 "ANSWER: <答案>"）
 */
async function sampleNode(state: typeof ConsistencyState.State) {
  const { question } = state;
  const N = 3;
  console.log(`\n🎲 [Sampler] 正在进行 ${N} 次独立推理...`);
  const promises = Array(N).fill(0).map((_, i) => model.invoke([new HumanMessage(`问题: ${question}\n请一步步思考，最后并在末尾单独一行输出: "ANSWER: <你的最终答案>"`)]));
  const results = await Promise.all(promises);
  return { samples: results.map(r => r.content as string) };
}

/**
 * 投票节点：分析所有样本，选择最一致、最正确的答案
 * 设计要点：
 * - 将所有样本作为上下文，让 LLM 进行对比分析
 * - 选择最一致且最正确的结论
 * - 可以识别并排除异常答案
 */
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
