/**
 * 智能体服务器
 * 提供 HTTP API 接口来调用各个智能体
 */

import "dotenv/config";
import express, { Request, Response } from "express";
import { app as parallelApp } from "../src/parallel_agent";

const server = express();
const PORT = process.env.PORT || 3000;

// 中间件：解析 JSON 请求体
server.use(express.json());

// 健康检查接口
server.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok", message: "智能体服务器运行中" });
});

// 并行智能体接口示例
server.post("/api/agents/parallel/start", async (req: Request, res: Response) => {
  try {
    const { topic } = req.body;
    
    if (!topic || typeof topic !== "string") {
      return res.status(400).json({ 
        error: "缺少必需参数: topic (string)" 
      });
    }

    console.log(`[API] 收到并行智能体请求: ${topic}`);
    const result = await parallelApp.invoke({ topic });
    
    res.json({
      success: true,
      data: {
        pros: result.pros,
        cons: result.cons,
        finalSummary: result.finalSummary
      }
    });
  } catch (error) {
    console.error("[API] 错误:", error);
    res.status(500).json({ 
      error: "智能体执行失败", 
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

// 启动服务器
server.listen(PORT, () => {
  console.log(`🚀 智能体服务器已启动，监听端口 ${PORT}`);
  console.log(`📡 健康检查: http://localhost:${PORT}/health`);
  console.log(`📡 并行智能体: POST http://localhost:${PORT}/api/agents/parallel/start`);
});
