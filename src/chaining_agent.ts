/**
 * 链式智能体 (Chaining Agent) / 流水线智能体
 * 
 * 功能概述：
 * 将复杂任务分解为多个顺序执行的步骤，每个步骤的输出作为下一步的输入。
 * 实现"分而治之"的流水线处理模式。
 * 
 * 设计要点：
 * 1. 顺序执行：严格按顺序执行，前一步的输出是后一步的输入
 * 2. 状态传递：通过 State 在节点间传递中间结果
 * 3. 任务分解：将复杂任务（如"写小说并翻译"）分解为简单步骤
 * 4. 工作流模式：Start -> Step1 -> Step2 -> Step3 -> End
 * 
 * 适用场景：
 * - 内容创作流水线（大纲 -> 初稿 -> 润色 -> 发布）
 * - 数据处理管道（提取 -> 清洗 -> 转换 -> 存储）
 * - 多阶段任务（规划 -> 执行 -> 验证 -> 交付）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：每个字段代表流水线的一个阶段
const PipelineState = Annotation.Root({
  topic: Annotation<string>,        // 输入主题
  outline: Annotation<string>,      // 阶段1输出：大纲
  draft: Annotation<string>,        // 阶段2输出：草稿
  finalOutput: Annotation<string>,  // 阶段3输出：最终结果
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 }); // 适中的创造性，适合创作任务

/**
 * 阶段1：生成大纲节点
 * 设计要点：接收原始主题，输出结构化的大纲
 */
async function outlineNode(state: typeof PipelineState.State) {
  const { topic } = state;
  console.log(`\n📑 [Step 1] 正在生成大纲: ${topic}`);
  const response = await model.invoke([new SystemMessage("你是一名小说家。请根据用户的主题，写一个包含3个章节的简短大纲。"), new HumanMessage(topic)]);
  console.log("生成大纲",response.content);
  return { outline: response.content as string };
}

/**
 * 阶段2：根据大纲扩写节点
 * 设计要点：使用上一步的 outline 作为输入，生成详细内容
 */
async function draftNode(state: typeof PipelineState.State) {
  const { outline } = state;
  console.log(`\n✍️ [Step 2] 正在根据大纲扩写...`);
  const response = await model.invoke([new SystemMessage("请根据提供的大纲，扩写成一篇500字以内的微小说。"), new HumanMessage(outline)]);
 
 console.log("生成微小说草稿",response.content);
  return { draft: response.content as string };
}

/**
 * 阶段3：翻译节点
 * 设计要点：使用上一步的 draft 作为输入，完成最终转换
 */
async function translateNode(state: typeof PipelineState.State) {
  const { draft } = state;
  console.log(`\n🌍 [Step 3] 正在翻译为英文...`);
  const response = await model.invoke([new SystemMessage("请将这篇小说翻译成优雅的英文。"), new HumanMessage(draft)]);
  return { finalOutput: response.content as string };
}

/**
 * 构建工作流图
 * 关键设计：严格的顺序执行，形成线性流水线
 * 每个节点必须等待前一个节点完成才能执行
 */
const workflow = new StateGraph(PipelineState)
  .addNode("generate_outline", outlineNode)   // 步骤1：生成大纲 (修改节点名以避免与状态字段冲突)
  .addNode("write_draft", draftNode)          // 步骤2：扩写草稿
  .addNode("translate", translateNode)        // 步骤3：翻译
  .addEdge("__start__", "generate_outline")   // 启动流程
  .addEdge("generate_outline", "write_draft") // 顺序连接
  .addEdge("write_draft", "translate")
  .addEdge("translate", END);                 // 完成

const app = workflow.compile();

async function main() {
  const input = { topic: "21世纪30年代人类重返月球考古阿波罗遗址，发现外星人遗迹" };
  const result = await app.invoke(input);
  console.log("\n====== 最终成果 (英文版) ======\n" + result.finalOutput);
}
main().catch(console.error);
