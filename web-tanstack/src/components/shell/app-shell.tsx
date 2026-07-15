import { Link, Outlet, useRouterState } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { Database, MessageSquareText, PanelLeftClose, PanelLeftOpen, PlugZap, Search, Sparkles } from "lucide-react"

import { api } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useUiStore } from "@/stores/ui-store"

const navItems = [
  { to: "/chat", label: "对话", icon: MessageSquareText },
  { to: "/documents", label: "文档", icon: Database },
  { to: "/retrieval", label: "检索", icon: Search },
  { to: "/mcp", label: "MCP", icon: PlugZap },
] as const

function HealthPill() {
  const query = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const result = await api.health()
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
    refetchInterval: 30_000,
    retry: 0,
  })

  const status =
    query.data?.status ?? (query.isError ? "offline" : query.isPending ? "checking" : "offline")
  const dotClass =
    status === "healthy"
      ? "bg-emerald-500"
      : status === "degraded"
        ? "bg-amber-500"
        : status === "offline"
          ? "bg-rose-500"
          : "bg-slate-400 animate-pulse"

  return (
    <div
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-slate-500"
      title={query.data ? `v${query.data.version}` : "健康检查"}
    >
      <span className={cn("size-2 rounded-full", dotClass)} />
      <span className="hidden sm:inline capitalize">{status}</span>
    </div>
  )
}

export function AppShell() {
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const sidebarCollapsed = useUiStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useUiStore((s) => s.toggleSidebar)

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white">
      {/* 顶部页眉 - 透明背景，类似 Gemini */}
      <header className="flex shrink-0 items-center justify-between gap-3 px-4 py-2.5">
        {/* 左侧：收起按钮 + Logo + 对话导航 */}
        <div className="flex items-center gap-2">
          <button
            className="inline-flex size-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900"
            onClick={toggleSidebar}
            title={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
            type="button"
          >
            {sidebarCollapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
          </button>

          <Link className="flex items-center gap-2" to="/chat" search={{}}>
            <div className="flex size-7 items-center justify-center rounded-lg bg-black text-white">
              <Sparkles className="size-3.5" />
            </div>
            <span
              className={cn(
                "text-sm font-semibold text-black transition-all duration-200 whitespace-nowrap",
                sidebarCollapsed ? "opacity-0 w-0 overflow-hidden" : "opacity-100 w-auto",
              )}
            >
              MindStack
            </span>
          </Link>

          {/* 对话导航 - 紧邻 Logo */}
          <nav className="ml-3 flex items-center gap-0.5">
            <Link
              key="/chat"
              to="/chat"
              search={{}}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm transition-colors",
                pathname.startsWith("/chat")
                  ? "bg-neutral-100 font-medium text-black"
                  : "text-neutral-500 hover:bg-neutral-50 hover:text-black",
              )}
            >
              <MessageSquareText className="size-3.5" />
              对话
            </Link>
          </nav>
        </div>

        {/* 右侧：文档 + 检索 + 健康状态 */}
        <div className="flex items-center gap-0.5">
          <nav className="flex items-center gap-0.5">
            {navItems.filter((item) => item.to !== "/chat").map((item) => {
              const isActive = pathname.startsWith(item.to)
              const Icon = item.icon
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  search={{}}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm transition-colors",
                    isActive
                      ? "bg-neutral-100 font-medium text-black"
                      : "text-neutral-500 hover:bg-neutral-50 hover:text-black",
                  )}
                >
                  <Icon className="size-3.5" />
                  {item.label}
                </Link>
              )
            })}
          </nav>
          <HealthPill />
        </div>
      </header>

      {/* 主区域 */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
