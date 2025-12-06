/**
 * 规划智能体 (Planning Agent)
 * 
 * 功能概述：
 * 先制定计划，然后循环执行计划中的每个步骤，最后整合所有结果。
 * 实现"规划-执行-整合"的智能任务处理模式。
 * 
 * 设计要点：
 * 1. 动态规划：使用 LLM 生成任务步骤，而非硬编码
 * 2. 循环执行：使用条件边实现循环，直到计划执行完毕
 * 3. 上下文累积：pastSteps 数组累积所有步骤的结果，供后续步骤参考
 * 4. 状态管理：plan 数组逐步减少，pastSteps 逐步增加
 * 5. 工作流模式：Start -> Planner -> Executor (循环) -> Responder -> End
 * 
 * 适用场景：
 * - 复杂任务分解（如"写论文"需要：研究 -> 大纲 -> 写作 -> 修改）
 * - 多步骤问题解决（如"搭建网站"需要：设计 -> 开发 -> 测试 -> 部署）
 * - 需要上下文传递的序列任务
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：目标、计划列表、已执行步骤、最终响应
const PlanningState = Annotation.Root({
  objective: Annotation<string>,                                                      // 用户目标
  plan: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),      // 计划步骤列表（逐步减少）
  pastSteps: Annotation<string[]>({ reducer: (x, y) => x.concat(y), default: () => [] }), // 已执行步骤结果（逐步增加）
  response: Annotation<string>,                                                       // 最终整合的响应
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 }); // temperature=0 确保计划生成的确定性

/**
 * 规划节点：根据目标生成执行计划
 * 设计要点：将复杂目标分解为可执行的步骤列表
 */
async function plannerNode(state: typeof PlanningState.State) {
  const { objective } = state;
  console.log(`\n📝 [Planner] 正在制定计划: "${objective}"...`);
  const prompt = `你是一个任务规划专家。\n目标: ${objective}\n请生成一个简短的步骤清单来实现这个目标。要求：最多 3-4 个步骤。返回格式必须是纯文本的列表，每行一个步骤。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const plan = response.content.toString().split('\n').filter(line => line.trim().length > 0);
  console.log(`📋 计划生成完毕，共 ${plan.length} 步。`);
  return { plan };
}

/**
 * 执行节点：执行计划中的当前步骤
 * 设计要点：
 * - 每次执行 plan[0]，执行后从 plan 中移除（plan.slice(1)）
 * - 将执行结果添加到 pastSteps，供后续步骤参考
 * - 使用 pastSteps 构建上下文，实现步骤间的信息传递
 */
async function executorNode(state: typeof PlanningState.State) {
  const { plan, pastSteps } = state;
  const currentStep = plan[0];
  console.log(`\n🔨 [Executor] 正在执行步骤: "${currentStep}"`);
  const context = pastSteps.map((s, i) => `步骤 ${i+1} 结果: ${s}`).join("\n");
  const prompt = `请执行以下任务: "${currentStep}"\n${context ? `这是之前的步骤产生的信息(供参考):\n${context}` : ""}\n请仅返回当前任务的执行结果。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const result = response.content as string;
  console.log(`✅ 步骤完成。结果预览: ${result.slice(0, 30)}...`);
  return { pastSteps: [result], plan: plan.slice(1) }; // 移除已执行步骤，添加结果
}

/**
 * 响应节点：整合所有步骤的结果，生成最终回复
 * 设计要点：基于所有 pastSteps 生成连贯的最终答案
 */
async function responseNode(state: typeof PlanningState.State) {
  console.log(`\n🎉 [Finalizer] 正在整合最终回复...`);
  const { objective, pastSteps } = state;
  const prompt = `用户目标: "${objective}"\n我们已经分步完成了所有任务，结果如下:\n${pastSteps.map((s, i) => `--- 步骤 ${i+1} ---\n${s}`).join("\n")}\n请基于以上信息，给用户一个连贯的、最终的回复。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { response: response.content as string };
}

/**
 * 循环控制函数：判断是否继续执行计划
 * 设计要点：这是实现循环的关键，根据 plan 长度决定下一步
 */
function shouldContinue(state: typeof PlanningState.State) {
  return state.plan.length > 0 ? "executor" : "responder";
}

/**
 * 构建工作流图
 * 关键设计：使用条件边实现循环执行
 * - executor 执行完后，如果 plan 还有剩余，继续执行 executor
 * - 如果 plan 为空，则进入 responder 生成最终响应
 */
const workflow = new StateGraph(PlanningState)
  .addNode("planner", plannerNode)          // 规划节点
  .addNode("executor", executorNode)         // 执行节点（可循环）
  .addNode("responder", responseNode)        // 响应节点
  .addEdge("__start__", "planner")          // 启动规划
  .addEdge("planner", "executor")           // 规划完成后开始执行
  .addConditionalEdges("executor", shouldContinue, {  // 条件循环
    executor: "executor",    // 如果还有计划，继续执行
    responder: "responder"   // 如果计划完成，生成响应
  })
  .addEdge("responder", END);               // 完成

const app = workflow.compile();

async function main() {
  const objective = "我想了解 Rust 语言的特点，并写一段 Hello World 代码解释其语法";
  const result = await app.invoke({ objective });
  console.log("\n====== FINAL OUTPUT ======\n" + result.response);
}
main().catch(console.error);
