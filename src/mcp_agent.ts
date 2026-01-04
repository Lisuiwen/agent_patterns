/**
 * MCP 客户端 / Agent 侧的最小演示：
 * - 读取类似 Cursor / CherryStudio 的 JSON 配置（mcp.config.json）
 * - 根据配置启动 MCP Server 进程（stdio）
 * - 走 MCP 握手：initialize
 * - 拉取工具：tools/list
 * - 调用工具：tools/call
 *
 * 你可以把它理解成“客户端把 MCP 工具动态注入 agent 的前半段”：
 * 1) JSON 配置告诉客户端要启动哪些 MCP Server
 * 2) 客户端与 server 建立 MCP 会话并拿到 tool schema（name/description/inputSchema）
 * 3) 客户端把 tool schema 映射成自己 agent 框架里的工具（例如 OpenAI function / LangChain tool）
 * 4) LLM 决定调用哪个工具 -> 客户端发 tools/call -> 把结果再喂回 LLM 继续推理
 */

import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import * as readline from "readline";
import { existsSync, readFileSync } from "fs";
import { join } from "path";

type McpServerConfig = {
  command: string;
  args?: string[];
  env?: Record<string, string>;
};

type McpConfigFile = {
  mcpServers: Record<string, McpServerConfig>;
};

type JsonRpcRequest = {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: unknown;
};

type JsonRpcResponse =
  | { jsonrpc: "2.0"; id: number; result: unknown }
  | { jsonrpc: "2.0"; id: number; error: { code: number; message: string; data?: unknown } };

const loadConfig = (configPath: string) => {
  const raw = readFileSync(configPath, "utf-8");
  const parsed = JSON.parse(raw) as McpConfigFile;
  if (!parsed?.mcpServers || typeof parsed.mcpServers !== "object") {
    throw new Error(`配置文件格式不正确：需要顶层字段 "mcpServers"，path=${configPath}`);
  }
  return parsed;
};

const resolveCommand = (command: string) => {
  // 如果已经是路径/带扩展名，就直接用
  const looksLikePath = command.includes("/") || command.includes("\\");
  if (looksLikePath) return command;

  // npm script 下，父进程能找到 tsx，但子进程 spawn('tsx') 可能找不到。
  // 这里显式去 node_modules/.bin 里找一份可执行文件。
  const binName = process.platform === "win32" ? `${command}.cmd` : command;
  const localBin = join(process.cwd(), "node_modules", ".bin", binName);
  if (existsSync(localBin)) return localBin;

  return command;
};

const spawnMcpServer = (cfg: McpServerConfig) => {
  const command = resolveCommand(cfg.command);
  const useShell = process.platform === "win32" && (command.toLowerCase().endsWith(".cmd") || command.toLowerCase().endsWith(".bat"));
  const child = spawn(command, cfg.args ?? [], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...(cfg.env ?? {}) },
    shell: useShell,
  });

  child.on("error", (error) => {
    console.error(`[MCP] 启动 server 失败：${String(error)}`);
  });

  child.on("exit", (code, signal) => {
    console.log(`[MCP] server 进程退出：code=${code} signal=${signal}`);
  });

  child.stderr.on("data", (buf) => {
    const text = buf.toString();
    if (text.trim()) console.error(`[MCP:stderr] ${text.trimEnd()}`);
  });

  return child;
};

class McpStdioClient {
  private child: ChildProcessWithoutNullStreams;
  private rl: readline.Interface;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void }>();

  constructor(child: ChildProcessWithoutNullStreams) {
    this.child = child;
    this.rl = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });
    this.rl.on("line", (line) => this.onLine(line));
  }

  private onLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let msg: JsonRpcResponse;
    try {
      msg = JSON.parse(trimmed) as JsonRpcResponse;
    } catch {
      console.error(`[MCP] 收到非 JSON 行：${trimmed}`);
      return;
    }

    const id = (msg as any)?.id;
    if (typeof id !== "number") return;

    const pending = this.pending.get(id);
    if (!pending) return;

    this.pending.delete(id);
    if ("error" in msg) {
      pending.reject(Object.assign(new Error(msg.error.message), { code: msg.error.code, data: msg.error.data }));
      return;
    }
    pending.resolve(msg.result);
  };

  request = async (method: string, params?: unknown) => {
    const id = this.nextId++;
    const req: JsonRpcRequest = { jsonrpc: "2.0", id, method, ...(params === undefined ? {} : { params }) };

    const payload = `${JSON.stringify(req)}\n`;
    this.child.stdin.write(payload);

    return await new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  };

  close = async () => {
    this.rl.close();
    this.child.kill();
  };
}

const pickFirstServer = (cfg: McpConfigFile) => {
  const names = Object.keys(cfg.mcpServers);
  if (names.length === 0) throw new Error("配置中没有任何 mcpServers");
  const name = names[0];
  return { name, server: cfg.mcpServers[name] };
};

const main = async () => {
  const configPath = process.env.MCP_CONFIG ?? join(process.cwd(), "mcp.config.json");
  const cfg = loadConfig(configPath);
  const { name, server } = pickFirstServer(cfg);

  console.log(`[MCP] 使用配置：${configPath}`);
  console.log(`[MCP] 启动 server="${name}"：${server.command} ${(server.args ?? []).join(" ")}`);

  const child = spawnMcpServer(server);
  const client = new McpStdioClient(child);

  try {
    const initResult = await client.request("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {
        // 这里只是演示；真实客户端会填更多能力（roots、sampling、resources 等）
        tools: {},
      },
      clientInfo: { name: "agent_patterns_demo_client", version: "0.0.1" },
    });
    console.log("[MCP] initialize result:", JSON.stringify(initResult, null, 2));

    const toolsList = await client.request("tools/list");
    console.log("[MCP] tools/list result:", JSON.stringify(toolsList, null, 2));

    // 演示：调用 math_add
    const call1 = await client.request("tools/call", { name: "math_add", arguments: { a: 7, b: 35 } });
    console.log("[MCP] tools/call math_add:", JSON.stringify(call1, null, 2));

    // 演示：调用 echo
    const call2 = await client.request("tools/call", { name: "echo", arguments: { text: "你好，MCP" } });
    console.log("[MCP] tools/call echo:", JSON.stringify(call2, null, 2));
  } finally {
    await client.close();
  }
};

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
