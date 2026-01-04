/**
 * 资源管理智能体 (Resource Agent) / 成本优化智能体
 * 
 * 功能概述：
 * 根据任务复杂度智能选择不同的处理模型，平衡成本和效果。
 * 简单任务使用低成本模型，复杂任务使用高性能模型。
 * 
 * 设计要点：
 * 1. 任务分类：使用 LLM 评估任务复杂度
 * 2. 资源路由：根据复杂度选择不同的处理节点
 * 3. 成本追踪：记录每次任务的成本
 * 4. 工作流模式：Start -> Classifier -> [Cheap | Expensive] -> End
 * 
 * 适用场景：
 * - 成本敏感的应用（需要控制 API 调用成本）
 * - 多模型系统（需要选择合适的模型）
 * - 资源优化（根据需求分配计算资源）
 * 
 * 扩展方向：
 * - 实现更细粒度的复杂度分类
 * - 添加成本预算和限制
 * - 支持模型性能监控和自动调整
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：任务、复杂度、成本、响应
const ResourceState = Annotation.Root({
  task: Annotation<string>,                    // 用户任务
  complexity: Annotation<"SIMPLE" | "COMPLEX">, // 任务复杂度
  cost: Annotation<number>,                    // 处理成本
  response: Annotation<string>,                // 最终响应
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const baseModel = new ChatOpenAI({ ...CONFIG, temperature: 0 }); // temperature=0 确保分类的确定性

/**
 * 分类节点：评估任务复杂度
 * 设计要点：
 * - 使用 LLM 进行任务分类（实际应用可使用更轻量的分类器）
 * - 返回标准化的复杂度标签
 */
async function classifierNode(state: typeof ResourceState.State) {
  const { task } = state;
  console.log(`\n⚖️ [Classifier] 正在评估任务复杂度: "${task}"`);
  const prompt = `请评估以下任务的复杂度。\n如果任务涉及简单的问候、翻译、事实查询，返回 "SIMPLE"。\n如果任务涉及逻辑推理、代码编写、创意写作，返回 "COMPLEX"。\n只返回一个单词。`;
  const res = await baseModel.invoke([new HumanMessage(prompt), new HumanMessage(task)]);
  const complexity = res.content.toString().includes("COMPLEX") ? "COMPLEX" : "SIMPLE";
  console.log(`   判定结果: ${complexity}`);
  return { complexity, cost: 0.1 };
}

/**
 * 低成本模型节点：处理简单任务
 * 设计要点：
 * - 使用简练的 SystemMessage，鼓励简短回答
 * - 成本较低（模拟）
 */
async function cheapModelNode(state: typeof ResourceState.State) {
  console.log(`\n⚡ [Flash Model] 使用高速低成本模型处理...`);
  const res = await baseModel.invoke([new SystemMessage("你是一个追求速度的助手。请用最简练的话回答。"), new HumanMessage(state.task)]);
  return { response: res.content as string, cost: 0.5 };
}

/**
 * 高成本模型节点：处理复杂任务
 * 设计要点：
 * - 使用详细的 SystemMessage，鼓励深入思考
 * - 成本较高（模拟）
 */
async function expensiveModelNode(state: typeof ResourceState.State) {
  console.log(`\n🐢 [Pro Model] 使用深度推理模型处理...`);
  const res = await baseModel.invoke([new SystemMessage("你是一个深度思考的专家。请详细、全面地回答，展示你的推理能力。"), new HumanMessage(state.task)]);
  return { response: res.content as string, cost: 10.0 };
}

/**
 * 路由逻辑：根据复杂度选择处理节点
 */
function routeLogic(state: typeof ResourceState.State) {
  return state.complexity === "COMPLEX" ? "expensive" : "cheap";
}

const workflow = new StateGraph(ResourceState)
  .addNode("classifier", classifierNode)
  .addNode("cheap", cheapModelNode)
  .addNode("expensive", expensiveModelNode)
  .addEdge("__start__", "classifier")
  .addConditionalEdges("classifier", routeLogic, { cheap: "cheap", expensive: "expensive" })
  .addEdge("cheap", END)
  .addEdge("expensive", END);

const app = workflow.compile();

import * as readline from "readline";

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let totalCost = 0;

  const promptUser = () =>
    new Promise<string>((resolve) => {
      rl.question("\n请输入你的任务（直接回车退出）：", (answer) => {
        resolve(answer.trim());
      });
    });

  while (true) {
    const task = await promptUser();
    if (!task) {
      break;
    }
    const res = await app.invoke({ task });
    console.log(`💬 回复: ${res.response.slice(0, 200)}\n💰 本次花费: ${res.cost}`);
    totalCost += res.cost;
  }

  console.log(`\n============== 总花费: ${totalCost} ==============`);

  rl.close();
}
main().catch(console.error);
