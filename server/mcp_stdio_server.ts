/**
 * 一个最小可用的 MCP Server（stdio 传输）。
 *
 * 目标：让“客户端通过 JSON 配置启动进程，然后 tools/list + tools/call”这条链路跑通。
 * 实现的 MCP 方法：
 * - initialize
 * - tools/list
 * - tools/call
 *
 * 说明：
 * - MCP 基于 JSON-RPC 2.0；stdio 传输通常是一行一个 JSON。
 * - 为了演示清晰，这里不引入额外依赖，手写一个很小的 router。
 */

import * as readline from "readline";

type JsonRpcRequest = {
  jsonrpc: "2.0";
  id?: string | number | null;
  method: string;
  params?: unknown;
};

type JsonRpcResponse =
  | { jsonrpc: "2.0"; id: string | number | null; result: unknown }
  | {
      jsonrpc: "2.0";
      id: string | number | null;
      error: { code: number; message: string; data?: unknown };
    };

const writeResponse = (res: JsonRpcResponse) => {
  process.stdout.write(`${JSON.stringify(res)}\n`);
};

const writeError = (id: string | number | null, code: number, message: string, data?: unknown) => {
  writeResponse({ jsonrpc: "2.0", id, error: { code, message, data } });
};

const PROTOCOL_VERSION = "2025-06-18";

const tools = [
  {
    name: "math_add",
    description: "把 a 与 b 相加并返回结果",
    inputSchema: {
      type: "object",
      properties: {
        a: { type: "number", description: "加数 a" },
        b: { type: "number", description: "加数 b" },
      },
      required: ["a", "b"],
    },
  },
  {
    name: "echo",
    description: "原样返回传入的 text",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string", description: "要回显的文本" },
      },
      required: ["text"],
    },
  },
] as const;

const handleInitialize = (id: string | number | null, params: unknown) => {
  const clientProtocol = (params as any)?.protocolVersion;
  const negotiated = typeof clientProtocol === "string" ? clientProtocol : PROTOCOL_VERSION;

  writeResponse({
    jsonrpc: "2.0",
    id,
    result: {
      protocolVersion: negotiated,
      capabilities: {
        tools: { listChanged: false },
      },
      serverInfo: {
        name: "demo-mcp-stdio-server",
        version: "0.0.1",
      },
    },
  });
};

const handleToolsList = (id: string | number | null) => {
  writeResponse({
    jsonrpc: "2.0",
    id,
    result: { tools },
  });
};

const handleToolsCall = (id: string | number | null, params: unknown) => {
  const name = (params as any)?.name;
  const args = (params as any)?.arguments ?? {};

  if (name === "math_add") {
    const a = (args as any)?.a;
    const b = (args as any)?.b;
    if (typeof a !== "number" || typeof b !== "number") {
      return writeError(id, -32602, "参数错误：math_add 需要 number 类型的 a、b", { name, arguments: args });
    }

    // MCP tools/call 返回结构：result.content 是一个数组（通常 text）。
    return writeResponse({
      jsonrpc: "2.0",
      id,
      result: {
        content: [{ type: "text", text: String(a + b) }],
      },
    });
  }

  if (name === "echo") {
    const text = (args as any)?.text;
    if (typeof text !== "string") {
      return writeError(id, -32602, "参数错误：echo 需要 string 类型的 text", { name, arguments: args });
    }
    return writeResponse({
      jsonrpc: "2.0",
      id,
      result: {
        content: [{ type: "text", text }],
      },
    });
  }

  return writeError(id, -32601, `未知工具：${String(name)}`, { name, arguments: args });
};

const handleRequest = (req: JsonRpcRequest) => {
  const id = req.id ?? null;

  // notification（无 id）不回包；演示里也简单忽略
  const hasId = typeof req.id === "string" || typeof req.id === "number" || req.id === null;
  const mustRespond = hasId && req.id !== undefined;

  try {
    if (req.jsonrpc !== "2.0") {
      if (mustRespond) writeError(id, -32600, "无效请求：jsonrpc 必须是 2.0");
      return;
    }

    if (req.method === "initialize") {
      if (mustRespond) handleInitialize(id, req.params);
      return;
    }

    if (req.method === "tools/list") {
      if (mustRespond) handleToolsList(id);
      return;
    }

    if (req.method === "tools/call") {
      if (mustRespond) handleToolsCall(id, req.params);
      return;
    }

    if (mustRespond) writeError(id, -32601, `未知方法：${req.method}`);
  } catch (error) {
    if (mustRespond) writeError(id, -32603, "服务器内部错误", error instanceof Error ? error.message : String(error));
  }
};

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    const req = JSON.parse(trimmed) as JsonRpcRequest;
    handleRequest(req);
  } catch (error) {
    // JSON 解析失败时无法确定 id，这里按 JSON-RPC 规范返回 id=null
    writeError(null, -32700, "解析错误：不是合法 JSON", error instanceof Error ? error.message : String(error));
  }
});

