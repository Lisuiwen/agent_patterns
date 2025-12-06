import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const PipelineState = Annotation.Root({
  topic: Annotation<string>,
  outline: Annotation<string>,
  draft: Annotation<string>,
  finalOutput: Annotation<string>,
});

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "kimi-k2-turbo-preview",
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.7 });

async function outlineNode(state: typeof PipelineState.State) {
  const { topic } = state;
  console.log(`\n📑 [Step 1] 正在生成大纲: ${topic}`);
  const response = await model.invoke([new SystemMessage("你是一名小说家。请根据用户的主题，写一个包含3个章节的简短大纲。"), new HumanMessage(topic)]);
  return { outline: response.content as string };
}

async function draftNode(state: typeof PipelineState.State) {
  const { outline } = state;
  console.log(`\n✍️ [Step 2] 正在根据大纲扩写...`);
  const response = await model.invoke([new SystemMessage("请根据提供的大纲，扩写成一篇500字以内的微小说。"), new HumanMessage(outline)]);
  return { draft: response.content as string };
}

async function translateNode(state: typeof PipelineState.State) {
  const { draft } = state;
  console.log(`\n🌍 [Step 3] 正在翻译为英文...`);
  const response = await model.invoke([new SystemMessage("请将这篇小说翻译成优雅的英文。"), new HumanMessage(draft)]);
  return { finalOutput: response.content as string };
}

const workflow = new StateGraph(PipelineState)
  .addNode("outline", outlineNode)
  .addNode("write_draft", draftNode)
  .addNode("translate", translateNode)
  .addEdge("__start__", "outline")
  .addEdge("outline", "write_draft")
  .addEdge("write_draft", "translate")
  .addEdge("translate", END);

const app = workflow.compile();

async function main() {
  const input = { topic: "一个时间旅行者回到古代教数学的故事" };
  const result = await app.invoke(input);
  console.log("\n====== 最终成果 (英文版) ======\n" + result.finalOutput);
}
main().catch(console.error);
