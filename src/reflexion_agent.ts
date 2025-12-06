import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

/**
 * ============================================================
 * 1. 定义大脑结构 (State Definition)
 * 生产级 Agent 需要精确的状态管理，而不仅仅是简单的 messages 数组
 * ============================================================
 */
const ReflexionState = Annotation.Root({
  // 用户的原始需求
  request: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "",
  }),
  
  // 当前生成的草稿内容
  content: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "",
  }),
  
  // 评分员的反馈意见
  critique: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "",
  }),
  
  // 当前迭代轮数 (用于防止死循环)
  revisionNumber: Annotation<number>({
    reducer: (x, y) => y,
    default: () => 0,
  }),
});

/**
 * ============================================================
 * 2. 初始化双模型 (Dual-Model Setup)
 * 技巧：生成者需要发散思维(temp=0.7)，评论者需要严谨逻辑(temp=0)
 * ============================================================
 */
const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};

// 👨‍🎨 生成者：负责写初稿和修改
const generatorModel = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

// 🕵️‍♂️ 评论家：负责挑刺
const criticModel = new ChatOpenAI({ ...CONFIG, temperature: 0 });

/**
 * ============================================================
 * 3. 定义核心节点 (Nodes)
 * ============================================================
 */

// 节点 A: 生成者 (Generator)
async function generationNode(state: typeof ReflexionState.State) {
  const { request, content, critique, revisionNumber } = state;
  
  console.log(`\n🤖 [Generator] 正在执行第 ${revisionNumber + 1} 版写作...`);

  let prompt = "";
  if (revisionNumber === 0) {
    // 初稿模式
    prompt = `你是一名专业的技术博主。
    用户请求: "${request}"
    
    请撰写一篇结构清晰、内容详实的初稿。只返回文章内容，不要其他废话。`;
  } else {
    // 修订模式
    prompt = `你是一名专业的技术博主。
    用户请求: "${request}"
    
    这是你之前的草稿:
    ---
    ${content}
    ---
    
    这是资深编辑给出的修改意见:
    "${critique}"
    
    请根据意见完全重写这篇文章。使其更完美。只返回新的文章内容。`;
  }

  const response = await generatorModel.invoke([new HumanMessage(prompt)]);

  return {
    content: response.content as string,
    revisionNumber: revisionNumber + 1
  };
}

// 节点 B: 评论家 (Critic)
async function reflectionNode(state: typeof ReflexionState.State) {
  const { request, content } = state;
  console.log(`\n🧐 [Critic] 正在评审草稿...`);

  const prompt = `你是一名极其严厉的资深技术编辑。你的目标是保证内容完美。
  
  用户原始请求: "${request}"
  
  当前草稿:
  ---
  ${content}
  ---
  
  请评审这篇草稿。
  1. 如果文章已经非常完美，完全符合要求，请直接仅回复: "TERMINATE"
  2. 否则，请列出 3 条具体的修改建议（建议应简明扼要）。`;

  const response = await criticModel.invoke([new HumanMessage(prompt)]);
  
  const critique = response.content as string;
  console.log(`📝 [意见]: ${critique.slice(0, 50)}...`); // 打印部分意见用于调试

  return { critique };
}

/**
 * ============================================================
 * 4. 路由逻辑 (Conditional Edges)
 * ============================================================
 */
const MAX_ITERATIONS = 3; // 生产环境必须设置最大重试次数

function shouldContinue(state: typeof ReflexionState.State) {
  const { critique, revisionNumber } = state;

  // 1. 熔断机制：防止无限循环浪费 Token
  if (revisionNumber >= MAX_ITERATIONS) {
    console.log("⚠️ [System] 达到最大重试次数，强制结束。");
    return END;
  }

  // 2. 质量达标：评论家说通过
  if (critique.includes("TERMINATE")) {
    console.log("✅ [System] 质量达标，通过评审。");
    return END;
  }

  // 3. 继续优化：回炉重造
  return "generate";
}

/**
 * ============================================================
 * 5. 组装图谱 (Graph Construction)
 * ============================================================
 */
const workflow = new StateGraph(ReflexionState)
  .addNode("generate", generationNode)
  .addNode("reflect", reflectionNode)
  
  .addEdge("__start__", "generate") // 启动 -> 写初稿
  .addEdge("generate", "reflect")   // 写完 -> 送审
  
  .addConditionalEdges("reflect", shouldContinue, {
    generate: "generate", // 意见不通过 -> 回去重写
    [END]: END            // 通过 -> 结束
  });

const app = workflow.compile();

/**
 * ============================================================
 * 6. 运行测试
 * ============================================================
 */
async function main() {
  const topic = "为什么网上很多人说阿波罗登月是假的";
  
  console.log(`🚀 开始 Reflexion 工作流，主题: ${topic}`);
  
  const inputs = {
    request: topic,
  };

  // 运行并获取最终状态
  const finalState = await app.invoke(inputs);
  
  console.log("\n==========================================");
  console.log("🎉 最终产出内容:");
  console.log("==========================================");
  console.log(finalState.content);
  console.log(`\n📊 统计: 共迭代 ${finalState.revisionNumber} 轮`);
}

main().catch(console.error);
