/**
 * 协作智能体 (Collaboration Agent) / 团队智能体
 * 
 * 功能概述：
 * 模拟团队协作模式：一个监督者（Supervisor）协调多个专业成员（Researcher、Writer），
 * 根据任务需求动态分配工作，实现分工协作。
 * 
 * 设计要点：
 * 1. 监督者模式：Supervisor 作为中央调度器，决定下一步行动
 * 2. 专业化分工：每个成员有特定角色和专长
 * 3. 动态路由：根据任务状态和需求动态选择下一个执行者
 * 4. 消息传递：通过 messages 数组在成员间传递工作成果
 * 5. 工作流模式：Start -> Supervisor -> [Researcher/Writer] -> Supervisor (循环) -> End
 * 
 * 适用场景：
 * - 内容创作团队（研究 -> 写作 -> 审核）
 * - 多专家咨询系统（不同领域专家协作）
 * - 复杂任务分解（需要多种技能）
 * 
 * 扩展方向：
 * - 添加更多专业角色（设计师、审核员等）
 * - 实现并行协作（多个成员同时工作）
 * - 添加任务优先级和资源管理
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage, BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// 定义状态：消息历史、下一个执行者
const TeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: (x, y) => x.concat(y), default: () => [] }), // 累积所有消息
  next: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "Supervisor" }),    // 下一个执行者（由 Supervisor 决定）
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 }); // 适中的创造性

/**
 * 研究员节点：负责搜集信息和数据
 * 设计要点：使用 SystemMessage 设定专业角色，确保回答的准确性
 */
async function researcherNode(state: typeof TeamState.State) {
  console.log("🕵️ [Researcher] 正在搜集信息...");
  const lastMessage = state.messages[state.messages.length - 1];
  const response = await model.invoke([new SystemMessage("你是一个研究员。请提供关于用户问题的准确数据。"), lastMessage]);
  return { messages: [response] };
}

/**
 * 作家节点：基于研究结果进行创作
 * 设计要点：使用最后一条消息（通常是研究结果）作为输入
 */
async function writerNode(state: typeof TeamState.State) {
  console.log("✍️ [Writer] 正在撰写文案...");
  const lastMessage = state.messages[state.messages.length - 1];
  const response = await model.invoke([new SystemMessage("你是一个作家。请基于之前的研究结果，写一段优美的文字。"), lastMessage]);
  return { messages: [response] };
}

/**
 * 监督者节点：分析当前状态，决定下一步行动
 * 设计要点：
 * - 查看所有历史消息，理解当前进度
 * - 根据任务需求决定调用 Researcher 还是 Writer
 * - 当任务完成时返回 "FINISH"
 */
async function supervisorNode(state: typeof TeamState.State) {
  console.log("👮 [Supervisor] 正在调度...");
  const { messages } = state;
  const systemPrompt = `你是一个团队管理者。团队成员有: "Researcher", "Writer"。\n规则:\n1. 如果用户的问题需要事实支撑，先让 "Researcher" 工作。\n2. 有了资料后，让 "Writer" 进行写作。\n3. 如果写作已完成且质量尚可，回复 "FINISH"。\n只返回一个单词: "Researcher", "Writer", 或 "FINISH"。`;
  const response = await model.invoke([new SystemMessage(systemPrompt), ...messages]);
  const decision = response.content.toString().trim().replace(/['"]/g, '');
  console.log(`👮 决策: ${decision}`);
  return { next: decision };
}

/**
 * 路由逻辑：根据 Supervisor 的决策路由到相应节点
 */
function routeLogic(state: typeof TeamState.State) {
  if (state.next === "Researcher") return "researcher";
  if (state.next === "Writer") return "writer";
  return END; // "FINISH" 或其他值则结束
}

/**
 * 构建工作流图
 * 关键设计：实现监督者循环模式
 * - Supervisor 始终是决策中心
 * - 成员完成任务后返回 Supervisor
 * - Supervisor 根据情况决定下一步或结束
 */
const workflow = new StateGraph(TeamState)
  .addNode("supervisor", supervisorNode)      // 监督者节点
  .addNode("researcher", researcherNode)      // 研究员节点
  .addNode("writer", writerNode)            // 作家节点
  .addEdge("__start__", "supervisor")       // 启动监督者
  .addConditionalEdges("supervisor", routeLogic, {  // 根据决策路由
    researcher: "researcher",
    writer: "writer",
    [END]: END
  })
  .addEdge("researcher", "supervisor")     // 成员完成后返回监督者
  .addEdge("writer", "supervisor");

const app = workflow.compile();

async function main() {
  const task = "请帮我写一段关于'量子计算'的简短介绍，风格要科幻一点。";
  console.log(`🚀 开始团队协作任务: ${task}`);
  const result = await app.invoke({ messages: [new HumanMessage(task)] });
  const lastMsg = result.messages[result.messages.length - 1];
  console.log("\n====== 最终产出 ======\n" + lastMsg.content);
}
main().catch(console.error);
