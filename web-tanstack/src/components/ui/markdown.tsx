import { memo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { cn } from "@/lib/utils"

/**
 * Markdown 渲染器
 *
 * 后端返回的 assistant 消息是 Markdown 文本（含代码块、表格、列表等），
 * 这里用 react-markdown + remark-gfm 渲染，并对常见元素做轻量样式适配。
 * 配色：极简白色风格，代码块用灰底黑字保证可读性。
 */
const baseComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  h1: (props) => <h1 className="mt-4 mb-2 text-xl font-semibold text-black" {...props} />,
  h2: (props) => <h2 className="mt-4 mb-2 text-lg font-semibold text-black" {...props} />,
  h3: (props) => <h3 className="mt-3 mb-2 text-base font-semibold text-black" {...props} />,
  h4: (props) => <h4 className="mt-3 mb-1 text-sm font-semibold text-neutral-900" {...props} />,
  p: (props) => <p className="my-2 leading-7 text-neutral-800" {...props} />,
  ul: (props) => <ul className="my-2 list-disc space-y-1 pl-6 text-neutral-800" {...props} />,
  ol: (props) => <ol className="my-2 list-decimal space-y-1 pl-6 text-neutral-800" {...props} />,
  li: (props) => <li className="leading-7 text-neutral-800" {...props} />,
  a: (props) => (
    <a
      className="text-black underline underline-offset-2 hover:text-neutral-600"
      target="_blank"
      rel="noreferrer noopener"
      {...props}
    />
  ),
  blockquote: (props) => (
    <blockquote
      className="my-2 border-l-4 border-neutral-300 bg-neutral-100 py-1 pl-3 text-neutral-600"
      {...props}
    />
  ),
  hr: () => <hr className="my-4 border-neutral-200" />,
  table: (props) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-neutral-200">
      <table className="w-full text-sm" {...props} />
    </div>
  ),
  thead: (props) => <thead className="bg-neutral-100 text-neutral-600" {...props} />,
  th: (props) => (
    <th className="border-b border-neutral-200 px-3 py-2 text-left font-medium" {...props} />
  ),
  td: (props) => <td className="border-b border-neutral-100 px-3 py-2 text-neutral-800" {...props} />,
  code: ({ className, children, ...props }) => {
    const isInline = !className?.includes("language-")
    if (isInline) {
      return (
        <code
          className="rounded px-1 font-mono text-[0.9em] text-neutral-900"
          {...props}
        >
          {children}
        </code>
      )
    }
    return (
      <code className={cn("font-mono text-[0.9em] text-black", className)} {...props}>
        {children}
      </code>
    )
  },
  pre: (props) => (
    <pre
      className="my-3 overflow-x-auto rounded-lg border border-neutral-200 bg-neutral-100 p-4 text-[13px] leading-6 text-black"
      {...props}
    />
  ),
}

export const Markdown = memo(function Markdown({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  if (!content) {
    return null
  }
  return (
    <div className={cn("text-sm text-neutral-800", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={baseComponents}>
        {content}
      </ReactMarkdown>
    </div>
  )
})
