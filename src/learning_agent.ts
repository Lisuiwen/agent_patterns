/**
 * 学习智能体 (Learning Agent) / 经验积累智能体
 * 
 * 功能概述：
 * 在执行任务后自动总结经验教训，并将经验应用到后续任务中。
 * 实现持续学习和改进的智能系统。
 * 
 * 设计要点：
 * 1. 经验检索：执行前从经验库中检索相关经验
 * 2. 经验应用：将经验作为 prompt 的一部分，指导任务执行
 * 3. 经验学习：任务完成后自动提取新经验
 * 4. 经验累积：新经验自动添加到经验库
 * 5. 工作流模式：Start -> Recall -> Act -> Learn -> End
 * 
 * 适用场景：
 * - 个性化助手（学习用户偏好）
 * - 持续改进系统（从错误中学习）
 * - 知识积累（逐步建立知识库）
 * 
 * 扩展方向：
 * - 使用向量数据库存储和检索经验
 * - 实现经验的重要性评分和淘汰机制
 * - 支持经验的版本管理和冲突解决
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import * as readline from "readline";

// 经验数据库文件路径（相对于项目根目录）
const EXPERIENCE_DB_PATH = join(process.cwd(), "assets/experience_db.json");

// 加载经验数据库
function loadExperienceDB(): string[] {
  try {
    const data = readFileSync(EXPERIENCE_DB_PATH, "utf-8");
    const parsed = JSON.parse(data);
    // 兼容两种格式：数组格式或对象格式 { "experiences": [...] }
    if (Array.isArray(parsed)) {
      return parsed;
    } else if (parsed && typeof parsed === "object" && Array.isArray(parsed.experiences)) {
      return parsed.experiences;
    } else {
      return [];
    }
  } catch (error) {
    // 如果文件不存在，返回空数组
    return [];
  }
}

// 保存经验数据库
function saveExperienceDB(experiences: string[]): void {
  try {
    // 确保目录存在
    const dir = dirname(EXPERIENCE_DB_PATH);
    mkdirSync(dir, { recursive: true });
    // 保存到文件
    writeFileSync(EXPERIENCE_DB_PATH, JSON.stringify(experiences, null, 2), "utf-8");
  } catch (error) {
    console.error("保存经验数据库失败:", error);
  }
}

// 经验数据库（从文件加载）
const EXPERIENCE_DB: string[] = loadExperienceDB();

// 定义状态：任务、检索到的经验、执行结果、新学到的经验
const LearningState = Annotation.Root({
  task: Annotation<string>,        // 用户任务
  retrievedContext: Annotation<string>,  // 检索到的经验
  result: Annotation<string>,      // 执行结果
  newInsight: Annotation<string>,  // 新学到的经验
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 }); // 适中的创造性

/**
 * 回忆节点：从经验库中检索相关经验
 * 设计要点：
 * - 实际应用应使用语义搜索匹配相关经验
 * - 这里简化实现，返回所有经验
 */
async function recallNode(state: typeof LearningState.State) {
  console.log(`\n📖 [Memory] 正在检索过往经验...`);
  const context = EXPERIENCE_DB.join("\n");
  return { retrievedContext: context };
}

/**
 * 执行节点：基于经验执行任务
 * 设计要点：
 * - 将检索到的经验作为 SystemMessage 的一部分
 * - 经验指导任务执行，确保遵循最佳实践
 */
async function actNode(state: typeof LearningState.State) {
  const { task, retrievedContext } = state;
  console.log(`\n✍️ [Actor] 正在执行任务...`);
  const prompt = `你是一个智能助手。请回答用户问题。\n⚠️ 重要：若与经验相关，务必根据过往经验回答。若完全不想管则可以自由发挥：\n${retrievedContext}\n用户任务: "${task}"`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  return { result: res.content as string };
}

/**
 * 学习节点：从任务执行中提取新经验
 * 设计要点：
 * - 分析任务和结果，提取通用经验
 * - 将新经验添加到经验库（实际应用应持久化）
 */
async function learnNode(state: typeof LearningState.State) {
  const { task, result } = state;
  console.log(`\n🧠 [Learner] 正在总结本次教训...`);
  const prompt = `任务: "${task}"\n回答: "${result}"\n请反思这次任务，提取用户的信息，总结成简短的一句话纳入经验。`;
  const res = await model.invoke([new HumanMessage(prompt)]);
  const insight = res.content as string;
  EXPERIENCE_DB.push(`新经验: ${insight}`);
  // 持久化保存到文件
  saveExperienceDB(EXPERIENCE_DB);
  console.log(`✅ 已通过学习获得新知识: "${insight}"`);
  return { newInsight: insight };
}

const workflow = new StateGraph(LearningState)
  .addNode("recall", recallNode)
  .addNode("act", actNode)
  .addNode("learn", learnNode)
  .addEdge("__start__", "recall")
  .addEdge("recall", "act")
  .addEdge("act", "learn")
  .addEdge("learn", END);

const app = workflow.compile();

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  // INSERT_YOUR_CODE
  while (true) {
    const task: string = await new Promise((resolve) => {
      rl.question("请输入你的任务（直接回车退出）: ", resolve);
    });
    if (!task.trim()) {
      rl.close();
      break;
    }
    const res = await app.invoke({ task });
    console.log("本次回复:", res.result);
    console.log("\n📚 当前经验库状态:", EXPERIENCE_DB);
  }
}
main().catch(console.error);
