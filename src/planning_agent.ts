import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const PlanningState = Annotation.Root({
  objective: Annotation<string>,
  plan: Annotation<string[]>({ reducer: (x, y) => y ?? x, default: () => [] }),
  pastSteps: Annotation<string[]>({ reducer: (x, y) => x.concat(y), default: () => [] }),
  response: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0 });

async function plannerNode(state: typeof PlanningState.State) {
  const { objective } = state;
  console.log(`\n📝 [Planner] 正在制定计划: "${objective}"...`);
  const prompt = `你是一个任务规划专家。\n目标: ${objective}\n请生成一个简短的步骤清单来实现这个目标。要求：最多 3-4 个步骤。返回格式必须是纯文本的列表，每行一个步骤。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const plan = response.content.toString().split('\n').filter(line => line.trim().length > 0);
  console.log(`📋 计划生成完毕，共 ${plan.length} 步。`);
  return { plan };
}

async function executorNode(state: typeof PlanningState.State) {
  const { plan, pastSteps } = state;
  const currentStep = plan[0];
  console.log(`\n🔨 [Executor] 正在执行步骤: "${currentStep}"`);
  const context = pastSteps.map((s, i) => `步骤 ${i+1} 结果: ${s}`).join("\n");
  const prompt = `请执行以下任务: "${currentStep}"\n${context ? `这是之前的步骤产生的信息(供参考):\n${context}` : ""}\n请仅返回当前任务的执行结果。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  const result = response.content as string;
  console.log(`✅ 步骤完成。结果预览: ${result.slice(0, 30)}...`);
  return { pastSteps: [result], plan: plan.slice(1) };
}

async function responseNode(state: typeof PlanningState.State) {
  console.log(`\n🎉 [Finalizer] 正在整合最终回复...`);
  const { objective, pastSteps } = state;
  const prompt = `用户目标: "${objective}"\n我们已经分步完成了所有任务，结果如下:\n${pastSteps.map((s, i) => `--- 步骤 ${i+1} ---\n${s}`).join("\n")}\n请基于以上信息，给用户一个连贯的、最终的回复。`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { response: response.content as string };
}

function shouldContinue(state: typeof PlanningState.State) {
  return state.plan.length > 0 ? "executor" : "responder";
}

const workflow = new StateGraph(PlanningState)
  .addNode("planner", plannerNode)
  .addNode("executor", executorNode)
  .addNode("responder", responseNode)
  .addEdge("__start__", "planner")
  .addEdge("planner", "executor")
  .addConditionalEdges("executor", shouldContinue, { executor: "executor", responder: "responder" })
  .addEdge("responder", END);

const app = workflow.compile();

async function main() {
  const objective = "我想了解 Rust 语言的特点，并写一段 Hello World 代码解释其语法";
  const result = await app.invoke({ objective });
  console.log("\n====== FINAL OUTPUT ======\n" + result.response);
}
main().catch(console.error);
