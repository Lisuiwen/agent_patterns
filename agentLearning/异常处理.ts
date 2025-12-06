import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const RobustState = Annotation.Root({
  task: Annotation<string>,
  attempts: Annotation<number>({ reducer: (x, y) => y, default: () => 0 }),
  errors: Annotation<string[]>({ reducer: (x, y) => x.concat(y), default: () => [] }),
  result: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.5 });

async function unstableToolNode(state: typeof RobustState.State) {
  const { attempts, task } = state;
  console.log(`\n⚡ [Primary Tool] 尝试第 ${attempts + 1} 次执行: "${task}"`);
  const isFailure = Math.random() > 0.2;
  if (isFailure && attempts < 2) {
    console.error("   ❌ 调用失败：网络超时或服务不可用。");
    return { attempts: attempts + 1, errors: [`Attempt ${attempts + 1}: Connection Timeout`] };
  }
  console.log("   ✅ 调用成功！");
  const response = await model.invoke([new SystemMessage("你是一个主处理单元。请处理用户任务。"), new HumanMessage(task)]);
  return { result: response.content as string, attempts: attempts + 1 };
}

async function fallbackNode(state: typeof RobustState.State) {
  const { task, errors } = state;
  console.log(`\n🛡️ [Fallback] 主节点多次失败，启用备用方案...\n   历史错误: ${errors.join(", ")}`);
  const prompt = `主系统已崩溃。你是一个备用系统 (Safe Mode)。请用最简短、最安全的方式回应用户任务: "${task}"\n并在开头注明 "[备用模式响应]"`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { result: response.content as string };
}

function routeLogic(state: typeof RobustState.State) {
  if (state.result) return END;
  if (state.attempts >= 3) return "fallback";
  return "primary_tool";
}

const workflow = new StateGraph(RobustState)
  .addNode("primary_tool", unstableToolNode)
  .addNode("fallback", fallbackNode)
  .addEdge("__start__", "primary_tool")
  .addConditionalEdges("primary_tool", routeLogic, { primary_tool: "primary_tool", fallback: "fallback", [END]: END })
  .addEdge("fallback", END);

const app = workflow.compile();

async function main() {
  console.log("🚀 开始任务：模拟不稳定环境...");
  const finalState = await app.invoke({ task: "分析 2024 年 Q3 财报数据" });
  console.log("\n====== 最终结果 ======\n" + finalState.result);
}
main().catch(console.error);

