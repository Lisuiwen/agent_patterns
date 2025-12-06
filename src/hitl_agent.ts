/**
 * 人机交互智能体 (Human-in-the-Loop Agent)
 * 
 * 功能概述：
 * 在关键决策点引入人工审核和反馈，实现人机协作的工作流。
 * AI 生成内容后，等待人类审核，根据反馈进行修改或批准。
 * 
 * 设计要点：
 * 1. 人工介入：在关键节点暂停，等待人类输入
 * 2. 反馈循环：根据反馈修改，直到获得批准
 * 3. 质量控制：通过人工审核确保输出质量
 * 4. 工作流模式：Start -> Write -> Human -> [Write (循环) | Send] -> End
 * 
 * 适用场景：
 * - 内容审核流程（AI 生成，人工审核）
 * - 重要决策支持（需要人工确认）
 * - 质量控制（确保输出符合标准）
 * - 敏感内容生成（需要人工把关）
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import * as readline from "readline";

// 定义状态：任务、草稿、人类反馈、最终结果
const HitlState = Annotation.Root({
  task: Annotation<string>,        // 原始任务
  draft: Annotation<string>,       // AI 生成的草稿
  feedback: Annotation<string>,    // 人类反馈（修改建议或 "approve"）
  finalResult: Annotation<string>, // 最终结果
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 }); // 适中的创造性

/**
 * 写作节点：根据任务生成草稿，或根据反馈修改草稿
 * 设计要点：
 * - 首次执行：根据 task 生成初始草稿
 * - 后续执行：根据 feedback 修改现有 draft
 * - 反馈处理：修改后将 feedback 清空，避免重复处理
 */
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

/**
 * 从终端读取用户输入
 */
function readUserInput(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

/**
 * 人类审核节点：等待真实人类输入反馈
 * 设计要点：
 * - 显示完整草稿内容供用户审核
 * - 等待用户在终端输入反馈
 * - 反馈格式：可以是修改建议（字符串）或 "approve"/"ok"（批准）
 */
async function humanReviewNode(state: typeof HitlState.State) {
  console.log("\n" + "=".repeat(60));
  console.log("🛑 [Human Review] 请审核以下草稿：");
  console.log("=".repeat(60));
  console.log(state.draft);
  console.log("=".repeat(60));
  
  const feedback = await readUserInput(
    "\n👤 请输入反馈（输入修改建议，或输入 'approve'/'ok' 批准）: "
  );

  if (feedback.toLowerCase() === "approve" || feedback.toLowerCase() === "ok") {
    console.log("✅ 已批准！");
    return { feedback: "approve" };
  } else if (feedback) {
    console.log(`📝 收到反馈: ${feedback}`);
    return { feedback };
  } else {
    // 如果用户直接回车，默认要求修改
    console.log("⚠️  未输入反馈，默认要求修改。");
    return { feedback: "请修改" };
  }
}

/**
 * 发送节点：最终批准后执行的操作
 * 设计要点：只有获得 "approve" 反馈后才能到达此节点
 */
async function sendNode(state: typeof HitlState.State) {
  console.log("\n📤 [Sender] 邮件已发送！(模拟)");
  return { finalResult: "SENT" };
}

/**
 * 路由逻辑：根据人类反馈决定下一步
 * 设计要点：
 * - "approve"：批准，进入发送流程
 * - 其他反馈：需要修改，返回 writer 节点
 */
function router(state: typeof HitlState.State) {
  if (state.feedback === "approve") return "sender";
  if (state.feedback) return "writer";
  return "sender";
}

/**
 * 构建工作流图
 * 关键设计：实现反馈循环
 * - writer -> human -> (根据反馈) -> writer (循环) 或 sender (批准)
 */
const hitlWorkflow = new StateGraph(HitlState)
  .addNode("writer", writeNode)              // 写作/修改节点
  .addNode("human", humanReviewNode)        // 人工审核节点（从终端读取输入）
  .addNode("sender", sendNode)              // 发送节点
  .addEdge("__start__", "writer")            // 启动写作
  .addEdge("writer", "human")                 // 写作完成后等待审核
  .addConditionalEdges("human", router, {     // 根据反馈路由
    writer: "writer",    // 需要修改，返回 writer
    sender: "sender"     // 已批准，进入发送
  })
  .addEdge("sender", END);                    // 完成

const app = hitlWorkflow.compile();
async function main() { await app.invoke({ task: "向老板请假去滑雪" }); }
main().catch(console.error);
