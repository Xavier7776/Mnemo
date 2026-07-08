import { create } from "zustand"

interface UiState {
  /** 当前激活的对话 ID */
  activeConversationId?: string
  setActiveConversationId: (value?: string) => void
  /** 会话侧边栏是否收起 */
  sidebarCollapsed: boolean
  setSidebarCollapsed: (value: boolean) => void
  toggleSidebar: () => void
}

export const useUiStore = create<UiState>((set) => ({
  activeConversationId: undefined,
  setActiveConversationId: (value) => set({ activeConversationId: value }),
  sidebarCollapsed: false,
  setSidebarCollapsed: (value) => set({ sidebarCollapsed: value }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}))
