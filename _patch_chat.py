"""在 routers/chat.py 中插入 tool_call 事件处理分支"""
import sys

fpath = r'd:\timeModel\advanced-rag\routers\chat.py'
with open(fpath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到 "result.get(\"type\") == \"complete\"" 这一行的行号
insert_line = None
for i, line in enumerate(lines):
    if 'result.get("type") == "complete"' in line and 'elif' in line:
        insert_line = i
        break

if insert_line is None:
    print("NOT FOUND")
    sys.exit(1)

# 要插入的代码（保持相同的缩进级别）
indent = '                        '  # 24 spaces = 6 levels of 4-space indent
new_lines = [
    f'{indent}elif result.get("type") == "tool_call":\n',
    f'{indent}    # 工具调用事件（前端可展示调用链）\n',
    f'{indent}    tool_call_data = {{\n',
    f'{indent}        "tool_call": {{\n',
    f'{indent}            "round": result.get("round", 1),\n',
    f'{indent}            "tools": result.get("tools", []),\n',
    f'{indent}        }}\n',
    f'{indent}    }\n',
    f'{indent}    data = json.dumps(tool_call_data, ensure_ascii=False)\n',
    f'{indent}    yield f"data: {{data}}\\n\\n"\n',
    f'{indent}\n',
]

# 在 complete 分支之前插入
lines = lines[:insert_line] + new_lines + lines[insert_line:]

with open(fpath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"OK - inserted at line {insert_line + 1}")
