/**
 * 多模态图片识别智能体 (Multimodal Vision Agent)
 * 
 * 功能概述：
 * 实现图片识别和分析功能，支持从本地文件读取图片，使用 Vision 模型理解图片内容并回答用户问题。
 * 能够识别物体、理解场景、分析细节、提取文字等。
 * 
 * 设计要点：
 * 1. 图片加载：从文件路径读取图片并转换为 base64 编码
 * 2. 多模态输入：使用 LangChain 的 HumanMessage 传递图片和文本
 * 3. Vision 模型：使用 Moonshot Vision API 进行图片理解
 * 4. 问答能力：基于图片内容回答用户问题
 * 5. 工作流模式：Start -> LoadImage -> AnalyzeImage -> End
 * 
 * 适用场景：
 * - 物体识别（"这张图片里有什么？"）
 * - 场景理解（"图片中的场景是什么？"）
 * - 细节分析（"图片中的人物在做什么？"）
 * - OCR 识别（"图片中的文字是什么？"）
 * - 图片描述生成
 * - 图片内容问答
 * 
 * 扩展方向：
 * - 支持多张图片同时分析
 * - 支持图片 URL 输入
 * - 添加图片预处理（裁剪、缩放等）
 * - 实现图片分类和标签生成
 */

import "dotenv/config";
import { Annotation, StateGraph, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { readFileSync, existsSync } from "fs";
import { extname } from "path";

// 定义状态：图片路径、用户问题、base64 编码的图片、分析结果
const MultimodalState = Annotation.Root({
  imagePath: Annotation<string>,                                                      // 图片文件路径
  question: Annotation<string>,                                                       // 用户问题
  imageBase64: Annotation<string>({ reducer: (x, y) => y ?? x, default: () => "" }), // base64 编码的图片
  answer: Annotation<string>,                                                         // 最终回答
});

// 支持的图片格式
const SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".webp"];

// 模型配置：使用 Moonshot Vision API
const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.moonshot.cn/v1" },
  modelName: "moonshot-v1-128k-vision-preview", // 使用 Vision 模型
};
const model = new ChatOpenAI({ ...CONFIG, temperature: 0.3 }); // 适中的 temperature 平衡准确性和创造性

/**
 * 获取图片的 MIME 类型
 * 设计要点：根据文件扩展名确定 MIME 类型，用于 base64 编码
 */
const getImageMimeType = (filePath: string): string => {
  const ext = extname(filePath).toLowerCase();
  const mimeTypes: Record<string, string> = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
  };
  return mimeTypes[ext] || "image/jpeg";
};

/**
 * 验证图片文件
 * 设计要点：
 * - 检查文件是否存在
 * - 验证文件格式是否支持
 * - 提供清晰的错误信息
 */
const validateImageFile = (filePath: string): void => {
  if (!existsSync(filePath)) {
    throw new Error(`图片文件不存在: ${filePath}`);
  }

  const ext = extname(filePath).toLowerCase();
  if (!SUPPORTED_FORMATS.includes(ext)) {
    throw new Error(
      `不支持的图片格式: ${ext}。支持的格式: ${SUPPORTED_FORMATS.join(", ")}`
    );
  }
};

/**
 * 图片加载节点：读取图片文件并转换为 base64 编码
 * 设计要点：
 * - 验证文件存在性和格式
 * - 读取文件并转换为 base64
 * - 生成 data URL 格式（data:image/jpeg;base64,...）
 */
const loadImageNode = async (state: typeof MultimodalState.State) => {
  const { imagePath } = state;
  console.log(`\n📷 [ImageLoader] 正在加载图片: "${imagePath}"`);

  try {
    // 验证文件
    validateImageFile(imagePath);

    // 读取文件并转换为 base64
    const imageBuffer = readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");
    const mimeType = getImageMimeType(imagePath);
    const dataUrl = `data:${mimeType};base64,${base64Image}`;

    console.log(`✅ 图片加载成功，大小: ${(imageBuffer.length / 1024).toFixed(2)} KB`);
    return { imageBase64: dataUrl };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`❌ 图片加载失败: ${errorMessage}`);
    throw error;
  }
};

/**
 * 图片分析节点：使用 Vision 模型分析图片并回答用户问题
 * 设计要点：
 * - 构建多模态消息（包含图片和文本）
 * - 使用 Vision 模型进行图片理解
 * - 基于图片内容回答用户问题
 */
const analyzeImageNode = async (state: typeof MultimodalState.State) => {
  const { imageBase64, question } = state;
  console.log(`\n🔍 [VisionAnalyzer] 正在分析图片...`);
  console.log(`❓ 用户问题: "${question}"`);

  try {
    // 构建多模态消息
    // LangChain 支持在 HumanMessage 的 content 中使用数组，包含文本和图片
    const message = new HumanMessage({
      content: [
        {
          type: "image_url",
          image_url: {
            url: imageBase64,
          },
        },
        {
          type: "text",
          text: question || "请详细描述这张图片的内容。",
        },
      ],
    });

    // 调用 Vision 模型
    const response = await model.invoke([message]);
    const answer = response.content as string;

    console.log(`✅ 分析完成`);
    return { answer };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`❌ 图片分析失败: ${errorMessage}`);
    throw error;
  }
};

// 构建工作流图
const workflow = new StateGraph(MultimodalState)
  .addNode("loadImage", loadImageNode)      // 图片加载节点
  .addNode("analyzeImage", analyzeImageNode) // 图片分析节点
  .addEdge("__start__", "loadImage")        // 启动时先加载图片
  .addEdge("loadImage", "analyzeImage")     // 加载完成后进行分析
  .addEdge("analyzeImage", END);            // 分析完成后结束

const app = workflow.compile();

/**
 * 主函数：演示多模态图片识别功能
 */
async function main() {
  // 示例：分析图片并回答不同的问题
  // 注意：需要提供实际的图片文件路径
  const testCases = [
    {
      imagePath: require("path").resolve(__dirname, "../assets/image.png"), // 使用绝对路径
      question: "把这张图变成黑白线稿图",
    },

  ];

  console.log("=".repeat(60));
  console.log("多模态图片识别智能体演示");
  console.log("=".repeat(60));

  for (const testCase of testCases) {
    try {
      console.log(`\n${"=".repeat(60)}`);
      console.log(`📸 图片: ${testCase.imagePath}`);
      console.log(`❓ 问题: ${testCase.question}`);
      console.log(`${"=".repeat(60)}`);

      const result = await app.invoke({
        imagePath: testCase.imagePath,
        question: testCase.question,
      });

      console.log(`\n💬 回答:\n${result.answer}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`\n❌ 处理失败: ${errorMessage}`);
      // 继续处理下一个测试用例
    }
  }
}

// 如果直接运行此文件，执行主函数
if (require.main === module) {
  main().catch((error) => {
    console.error("程序执行出错:", error);
    process.exitCode = 1;
  });
}

export { app, MultimodalState };
