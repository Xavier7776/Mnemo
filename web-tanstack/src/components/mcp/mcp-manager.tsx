/** MCP 管理页面 — Server 状态、快捷添加、工具查看 */

import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  Pencil,
  PlugZap,
  Plus,
  RefreshCw,
  Trash2,
  Wrench,
  X,
  Zap,
} from "lucide-react"

import { api } from "@/lib/api"
import { cn } from "@/lib/utils"
import type {
  McpServerConfig,
  McpServerStatus,
  McpStatus,
  McpToolsListResponse,
} from "@/types/api"

// —— 预设快捷模板 ——
const QUICK_TEMPLATES: Record<string, { name: string; config: McpServerConfig }> = {
  fetch: {
    name: "fetch",
    config: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-fetch"],
      env: {},
      enabled: true,
      timeout: 30,
    },
  },
  filesystem: {
    name: "filesystem",
    config: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"],
      env: {},
      enabled: true,
      timeout: 30,
    },
  },
  github: {
    name: "github",
    config: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
      env: { GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}" },
      enabled: true,
      timeout: 30,
    },
  },
  sqlite: {
    name: "sqlite",
    config: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./data.db"],
      env: {},
      enabled: true,
      timeout: 30,
    },
  },
}

const EMPTY_FORM: { name: string; config: McpServerConfig } = {
  name: "",
  config: {
    transport: "stdio",
    command: "",
    args: [],
    env: {},
    enabled: true,
    timeout: 30,
  },
}

export function McpManager() {
  const queryClient = useQueryClient()
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingServer, setEditingServer] = useState<string | null>(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [expandedServer, setExpandedServer] = useState<string | null>(null)
  const [removeTarget, setRemoveTarget] = useState<string | null>(null)
  const [argsText, setArgsText] = useState("")
  const [envText, setEnvText] = useState("")
  const [showJsonImport, setShowJsonImport] = useState(false)
  const [jsonImportText, setJsonImportText] = useState("")
  const [jsonImportError, setJsonImportError] = useState<string | null>(null)
  const [jsonImportProgress, setJsonImportProgress] = useState<{ current: number; total: number; currentName: string } | null>(null)
  const [jsonImportSuccess, setJsonImportSuccess] = useState<string | null>(null)

  // 查询 MCP 状态
  const statusQuery = useQuery({
    queryKey: ["mcp", "status"],
    queryFn: async () => {
      const result = await api.getMcpStatus()
      if (result.error) throw new Error(result.error)
      return result.data as McpStatus
    },
    refetchInterval: 5_000,
    retry: 1,
  })

  // 工具列表（按需懒加载）
  const toolsQuery = useQuery({
    queryKey: ["mcp", "tools"],
    queryFn: async () => {
      const result = await api.listMcpTools()
      if (result.error) throw new Error(result.error)
      return result.data as McpToolsListResponse
    },
    enabled: !!expandedServer,
    retry: 1,
  })

  // 变更操作
  const addMutation = useMutation({
    mutationFn: async ({ name, config }: { name: string; config: McpServerConfig }) => {
      const result = await api.addMcpServer(name, config)
      if (result.error) throw new Error(result.error)
      return result.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp"] })
      resetForm()
    },
  })

  const updateMutation = useMutation({
    mutationFn: async ({ name, config }: { name: string; config: McpServerConfig }) => {
      const result = await api.updateMcpServer(name, config)
      if (result.error) throw new Error(result.error)
      return result.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp"] })
      resetForm()
    },
  })

  const removeMutation = useMutation({
    mutationFn: async (name: string) => {
      const result = await api.removeMcpServer(name)
      if (result.error) throw new Error(result.error)
      return result.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp"] })
    },
  })

  const reconnectMutation = useMutation({
    mutationFn: async (name: string) => {
      const result = await api.reconnectMcpServer(name)
      if (result.error) throw new Error(result.error)
      return result.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp"] })
    },
  })

  async function handleJsonImport() {
    setJsonImportError(null)
    setJsonImportSuccess(null)

    let parsed: unknown
    try {
      parsed = JSON.parse(jsonImportText)
    } catch (e) {
      setJsonImportError(`JSON 解析失败: ${(e as Error).message}`)
      return
    }

    const servers = (parsed as { mcpServers?: Record<string, unknown>; servers?: Record<string, unknown> }).mcpServers
      ?? (parsed as { servers?: Record<string, unknown> }).servers
      ?? {}
    const names = Object.keys(servers)
    if (names.length === 0) {
      setJsonImportError("未找到 mcpServers 配置")
      return
    }

    // 逐个导入，显示进度
    const results: { name: string; ok: boolean; error?: string }[] = []
    for (let i = 0; i < names.length; i++) {
      const name = names[i]
      setJsonImportProgress({ current: i + 1, total: names.length, currentName: name })

      const s = servers[name] as Record<string, unknown>

      // 直接透传 args，不做平台路径检测——由后端负责启动与错误反馈
      // 这样用户可以直接粘贴官方文档的 JSON 配置（含 npx -y 和任意路径）
      const args = Array.isArray(s.args) ? s.args as string[] : []

      const transportRaw = (s.transport as string) ?? "stdio"
      const config: McpServerConfig = {
        transport: transportRaw === "sse" ? "sse" : "stdio",
        command: (s.command as string) ?? "",
        args,
        env: (s.env as Record<string, string>) ?? {},
        url: (s.url as string | null) ?? null,
        enabled: s.enabled !== false,
        timeout: (s.timeout as number) ?? 30,
      }

      const result = await api.addMcpServer(name, config)
      if (result.error) {
        console.error(`[MCP导入] ${name} 失败:`, result.error)
        results.push({ name, ok: false, error: result.error })
      } else {
        results.push({ name, ok: true })
      }
    }

    setJsonImportProgress(null)
    await queryClient.invalidateQueries({ queryKey: ["mcp"] })

    const successCount = results.filter((r) => r.ok).length
    const failedCount = results.length - successCount

    if (failedCount === 0) {
      setJsonImportSuccess(`导入完成：${successCount} 个 Server 全部成功`)
      setJsonImportText("")
      // 2 秒后自动关闭导入面板
      setTimeout(() => {
        setShowJsonImport(false)
        setJsonImportSuccess(null)
      }, 2000)
    } else {
      const failedNames = results.filter((r) => !r.ok).map((r) => `${r.name}: ${r.error}`).join("\n")
      setJsonImportError(`导入完成：${successCount} 成功，${failedCount} 失败\n${failedNames}`)
    }

    console.log(`[MCP导入] 完成，成功 ${successCount}/${names.length}`)
  }

  function resetForm() {
    setShowAddForm(false)
    setEditingServer(null)
    setForm(EMPTY_FORM)
    setArgsText("")
    setEnvText("")
  }

  function applyTemplate(key: string) {
    const tpl = QUICK_TEMPLATES[key]
    setForm({ name: tpl.name, config: { ...tpl.config } })
    setArgsText(tpl.config.args?.join("\n") ?? "")
    setEnvText(
      Object.entries(tpl.config.env ?? {})
        .map(([k, v]) => `${k}=${v}`)
        .join("\n"),
    )
    setEditingServer(null)
    setShowAddForm(true)
  }

  function startEdit(server: McpServerStatus) {
    setForm({
      name: server.name,
      config: {
        transport: server.transport as "stdio" | "sse",
        command: "",
        args: [],
        env: {},
        url: null,
        enabled: server.enabled,
        timeout: server.timeout,
      },
    })
    setArgsText("")
    setEnvText("")
    setEditingServer(server.name)
    setShowAddForm(true)
  }

  function handleSubmit() {
    const config: McpServerConfig = {
      ...form.config,
      args: argsText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      env: envText
        .split("\n")
        .map((line) => {
          const idx = line.indexOf("=")
          if (idx < 0) return null
          return [line.slice(0, idx).trim(), line.slice(idx + 1).trim()] as const
        })
        .filter((v): v is readonly [string, string] => v !== null)
        .reduce<Record<string, string>>((acc, [k, v]) => {
          acc[k] = v
          return acc
        }, {}),
    }

    if (editingServer) {
      updateMutation.mutate({ name: editingServer, config })
    } else {
      addMutation.mutate({ name: form.name, config })
    }
  }

  const status = statusQuery.data
  const isPending = statusQuery.isPending
  const isError = statusQuery.isError

  return (
    <div className="mx-auto h-full w-full max-w-5xl space-y-4 overflow-y-auto px-4 py-6">
      {/* 标题栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <PlugZap className="size-5 text-blue-500" />
          <h1 className="text-lg font-semibold text-slate-800">MCP 管理</h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowJsonImport(!showJsonImport)}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-600 transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-600"
          >
            <Plus className="size-4" />
            导入 JSON
          </button>
          <button
            onClick={() => {
              resetForm()
              setShowAddForm(true)
            }}
            className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-600"
          >
            <Plus className="size-4" />
            添加 Server
          </button>
        </div>
      </div>

      {/* 状态条 */}
      {isPending ? (
        <div className="flex items-center gap-2 rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-500">
          <Loader2 className="size-4 animate-spin" />
          正在加载 MCP 状态...
        </div>
      ) : isError ? (
        <div className="rounded-xl border-l-4 border-rose-400 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          MCP 服务未响应。请确认后端已设置 MCP_ENABLED=true 并重启。
        </div>
      ) : status ? (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
          <StatusBadge label="状态" value={status.enabled ? "已启用" : "未启用"} ok={status.enabled} />
          <StatusBadge
            label="连接"
            value={`${status.connected_servers}/${status.total_servers}`}
            ok={status.connected_servers > 0}
          />
          <StatusBadge label="工具" value={String(status.total_tools)} ok={status.total_tools > 0} neutral />
          {status.compact_mode && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
              紧凑模式
            </span>
          )}
          <span className="ml-auto text-xs text-slate-400">
            {status.initialized ? "已初始化" : "未初始化"}
          </span>
        </div>
      ) : null}

      {/* JSON 导入面板 */}
      {showJsonImport && (
        <div className="rounded-xl border border-blue-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-700">导入 Claude Desktop 风格 JSON 配置</h2>
            <button onClick={() => setShowJsonImport(false)} className="text-slate-400 hover:text-slate-600">
              <X className="size-4" />
            </button>
          </div>
          <p className="mb-2 text-xs text-slate-500">
            粘贴 Claude Desktop 格式的 JSON，支持 <code className="rounded bg-slate-100 px-1">{"{ mcpServers: { ... } }"}</code> 或 <code className="rounded bg-slate-100 px-1">{"{ servers: { ... } }"}</code>。每个 Server 支持 command / args / env 字段。
          </p>

          {jsonImportProgress ? (
            <div className="mb-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2">
              <div className="mb-1.5 flex items-center justify-between text-xs text-blue-700">
                <span className="flex items-center gap-1.5">
                  <Loader2 className="size-3.5 animate-spin" />
                  正在导入 {jsonImportProgress.currentName}...
                </span>
                <span>{jsonImportProgress.current} / {jsonImportProgress.total}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full bg-blue-500 transition-all"
                  style={{ width: `${(jsonImportProgress.current / jsonImportProgress.total) * 100}%` }}
                />
              </div>
            </div>
          ) : null}

          {jsonImportSuccess ? (
            <div className="mb-3 rounded-lg border-l-4 border-emerald-400 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              {jsonImportSuccess}
            </div>
          ) : null}

          {jsonImportError ? (
            <div className="mb-3 whitespace-pre-line rounded-lg border-l-4 border-rose-400 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {jsonImportError}
            </div>
          ) : null}

          <textarea
            className="h-48 w-full rounded-lg border border-slate-200 p-3 font-mono text-xs text-slate-700 outline-none focus:border-blue-400 disabled:bg-slate-50"
            placeholder={'{\n  "mcpServers": {\n    "example-server": {\n      "command": "npx",\n      "args": ["-y", "mcp-server-example"],\n      "env": {}\n    }\n  }\n}'}
            value={jsonImportText}
            onChange={(e) => setJsonImportText(e.target.value)}
            disabled={!!jsonImportProgress}
          />

          <div className="mt-3 flex justify-end gap-2">
            <button
              onClick={() => {
                setJsonImportText("")
                setJsonImportError(null)
                setJsonImportSuccess(null)
                setShowJsonImport(false)
              }}
              disabled={!!jsonImportProgress}
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-500 hover:bg-slate-50 disabled:opacity-40"
            >
              取消
            </button>
            <button
              onClick={handleJsonImport}
              disabled={!jsonImportText.trim() || !!jsonImportProgress}
              className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-600 disabled:opacity-40"
            >
              {jsonImportProgress ? <Loader2 className="size-4 animate-spin" /> : null}
              {jsonImportProgress ? "导入中..." : "导入"}
            </button>
          </div>
        </div>
      )}

      {/* 快捷模板 */}
      {!showAddForm && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="mb-2 text-xs font-medium text-slate-500">快捷添加（一键预填配置）</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(QUICK_TEMPLATES).map(([key, tpl]) => (
              <button
                key={key}
                onClick={() => applyTemplate(key)}
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-600 transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-600"
              >
                <Zap className="size-3.5 text-blue-400" />
                {tpl.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 添加/编辑表单 */}
      {showAddForm && (
        <div className="rounded-xl border border-blue-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-700">
              {editingServer ? `编辑 ${editingServer}` : "添加新 Server"}
            </h2>
            <button onClick={resetForm} className="text-slate-400 hover:text-slate-600">
              <X className="size-4" />
            </button>
          </div>

          <div className="space-y-3">
            {/* 名称 */}
            {!editingServer && (
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">Server 名称</label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="如: filesystem"
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
                />
              </div>
            )}

            {/* 传输方式 */}
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-500">传输方式</label>
              <div className="flex gap-2">
                {(["stdio", "sse"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setForm({ ...form, config: { ...form.config, transport: t } })}
                    className={cn(
                      "rounded-lg border px-4 py-1.5 text-sm transition-colors",
                      form.config.transport === t
                        ? "border-blue-400 bg-blue-50 text-blue-600"
                        : "border-slate-200 text-slate-500 hover:border-slate-300",
                    )}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {/* stdio 字段 */}
            {form.config.transport === "stdio" && (
              <>
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-500">命令 (command)</label>
                  <input
                    type="text"
                    value={form.config.command ?? ""}
                    onChange={(e) =>
                      setForm({ ...form, config: { ...form.config, command: e.target.value } })
                    }
                    placeholder="如: npx 或 python"
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-500">
                    参数 (args，每行一个)
                  </label>
                  <textarea
                    value={argsText}
                    onChange={(e) => setArgsText(e.target.value)}
                    placeholder={"如:\n-y\n@modelcontextprotocol/server-fetch"}
                    rows={4}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs outline-none focus:border-blue-400"
                  />
                </div>
              </>
            )}

            {/* sse 字段 */}
            {form.config.transport === "sse" && (
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">URL</label>
                <input
                  type="text"
                  value={form.config.url ?? ""}
                  onChange={(e) =>
                    setForm({ ...form, config: { ...form.config, url: e.target.value } })
                  }
                  placeholder="如: http://localhost:8081/sse"
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
                />
              </div>
            )}

            {/* 环境变量 */}
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-500">
                环境变量 (每行 KEY=VALUE，支持 $&#123;VAR&#125; 插值)
              </label>
              <textarea
                value={envText}
                onChange={(e) => setEnvText(e.target.value)}
                placeholder={"如:\nGITHUB_PERSONAL_ACCESS_TOKEN=${GITHUB_TOKEN}"}
                rows={3}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs outline-none focus:border-blue-400"
              />
            </div>

            {/* timeout + enabled */}
            <div className="flex items-center gap-4">
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">超时 (秒)</label>
                <input
                  type="number"
                  value={form.config.timeout}
                  onChange={(e) =>
                    setForm({ ...form, config: { ...form.config, timeout: Number(e.target.value) } })
                  }
                  className="w-24 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
                />
              </div>
              <label className="flex items-center gap-2 pt-5 text-sm text-slate-600">
                <input
                  type="checkbox"
                  checked={form.config.enabled}
                  onChange={(e) =>
                    setForm({ ...form, config: { ...form.config, enabled: e.target.checked } })
                  }
                  className="size-4 accent-blue-500"
                />
                启用
              </label>
            </div>

            {/* 操作按钮 */}
            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={handleSubmit}
                disabled={addMutation.isPending || updateMutation.isPending || (!editingServer && !form.name)}
                className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-600 disabled:opacity-50"
              >
                {(addMutation.isPending || updateMutation.isPending) && (
                  <Loader2 className="size-4 animate-spin" />
                )}
                {editingServer ? "保存" : "添加"}
              </button>
              <button
                onClick={resetForm}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm text-slate-500 hover:bg-slate-50"
              >
                取消
              </button>
              {(addMutation.error || updateMutation.error) && (
                <span className="text-xs text-rose-500">
                  {((addMutation.error || updateMutation.error) as Error).message}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Server 列表 */}
      <div className="space-y-2">
        {status?.servers.length === 0 && !isPending ? (
          <div className="rounded-xl border border-dashed border-slate-200 bg-white py-12 text-center text-sm text-slate-400">
            暂无 MCP Server，点击上方按钮添加
          </div>
        ) : (
          status?.servers.map((server) => (
            <ServerCard
              key={server.name}
              server={server}
              expanded={expandedServer === server.name}
              onToggle={() =>
                setExpandedServer(expandedServer === server.name ? null : server.name)
              }
              tools={toolsQuery.data?.servers?.[server.name] ?? []}
              toolsLoading={toolsQuery.isPending && expandedServer === server.name}
              onReconnect={() => reconnectMutation.mutate(server.name)}
              onEdit={() => startEdit(server)}
              onRemove={() => setRemoveTarget(server.name)}
              reconnecting={reconnectMutation.isPending && reconnectMutation.variables === server.name}
              removing={removeMutation.isPending && removeMutation.variables === server.name}
            />
          ))
        )}
      </div>

      {/* 删除确认弹窗 */}
      {removeTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setRemoveTarget(null)}>
          <div
            className="w-full max-w-sm rounded-2xl bg-white p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="mb-2 text-base font-semibold text-slate-800">确认移除</h3>
            <p className="mb-4 text-sm text-slate-500">
              确认移除 Server <span className="font-medium text-slate-700">"{removeTarget}"</span>？该操作会断开连接并删除配置。
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setRemoveTarget(null)}
                disabled={removeMutation.isPending}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm text-slate-500 hover:bg-slate-50 disabled:opacity-40"
              >
                取消
              </button>
              <button
                onClick={() => {
                  removeMutation.mutate(removeTarget, {
                    onSuccess: () => setRemoveTarget(null),
                  })
                }}
                disabled={removeMutation.isPending}
                className="flex items-center gap-1.5 rounded-lg bg-rose-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-rose-600 disabled:opacity-40"
              >
                {removeMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
                确认移除
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

// —— 子组件 ——

function StatusBadge({
  label,
  value,
  ok,
  neutral,
}: {
  label: string
  value: string
  ok?: boolean
  neutral?: boolean
}) {
  const color = neutral
    ? "bg-slate-100 text-slate-600"
    : ok
      ? "bg-emerald-50 text-emerald-700"
      : "bg-rose-50 text-rose-700"
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-slate-400">{label}</span>
      <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", color)}>{value}</span>
    </div>
  )
}

function ServerCard({
  server,
  expanded,
  onToggle,
  tools,
  toolsLoading,
  onReconnect,
  onEdit,
  onRemove,
  reconnecting,
  removing,
}: {
  server: McpServerStatus
  expanded: boolean
  onToggle: () => void
  tools: Array<{ name: string; description: string; parameters: Record<string, unknown> }>
  toolsLoading: boolean
  onReconnect: () => void
  onEdit: () => void
  onRemove: () => void
  reconnecting: boolean
  removing: boolean
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      {/* 行头 */}
      <div className="flex items-center gap-3 px-4 py-3">
        <button onClick={onToggle} className="text-slate-400 hover:text-slate-600">
          {expanded ? (
            <ChevronDown className="size-4" />
          ) : (
            <ChevronRight className="size-4" />
          )}
        </button>

        {/* 状态灯 */}
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            server.connected ? "bg-emerald-500" : "bg-rose-400",
          )}
        />

        {/* 名称 + 传输 */}
        <div className="flex-1">
          <span className="font-medium text-slate-700">{server.name}</span>
          <span className="ml-2 text-xs text-slate-400">{server.transport}</span>
        </div>

        {/* 工具数 */}
        <span className="text-xs text-slate-400">{server.tool_count} 个工具</span>

        {/* 操作按钮 */}
        <button
          onClick={onReconnect}
          disabled={reconnecting}
          title="重连"
          className="rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-blue-500 disabled:opacity-50"
        >
          {reconnecting ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
        </button>
        <button
          onClick={onEdit}
          title="编辑"
          className="rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-blue-500"
        >
          <Pencil className="size-4" />
        </button>
        <button
          onClick={onRemove}
          disabled={removing}
          title="移除"
          className="rounded-md p-1.5 text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-500 disabled:opacity-50"
        >
          {removing ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
        </button>
      </div>

      {/* 展开内容：工具列表 */}
      {expanded && (
        <div className="max-h-[400px] overflow-y-auto border-t border-slate-100 bg-slate-50 px-4 py-3">
          {toolsLoading ? (
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Loader2 className="size-4 animate-spin" />
              正在加载工具详情...
            </div>
          ) : tools.length === 0 ? (
            <div className="text-sm text-slate-400">该 Server 无工具</div>
          ) : (
            <div className="space-y-2">
              {tools.map((tool) => (
                <div key={tool.name} className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="flex items-center gap-2">
                    <Wrench className="size-3.5 shrink-0 text-slate-400" />
                    <span className="font-mono text-sm font-medium text-slate-700">
                      {tool.name}
                    </span>
                  </div>
                  <p className="mt-1 pl-5.5 text-xs text-slate-500">{tool.description}</p>
                  {Object.keys(tool.parameters).length > 0 && (
                    <pre className="mt-2 overflow-x-auto rounded bg-slate-100 p-2 text-xs text-slate-600">
                      {JSON.stringify(tool.parameters, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
