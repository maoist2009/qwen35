#!/bin/bash

# 检查是否传入了参数
if [ -z "$1" ]; then
    echo "错误: 请提供一个新的 API Key 作为参数。"
    echo "用法: $0 <new_api_key>"
    exit 1
fi

NEW_KEY="$1"
FILE1="$HOME/.qwen/settings.json"
FILE2="$HOME/.qwen/settings.json.orig"

# 定义替换函数
update_json_key() {
    local file="$1"
    if [ -f "$file" ]; then
        # 使用 sed 精确匹配并替换 OPENROUTER_API_KEY 的值
        sed -i 's/"OPENROUTER_API_KEY": "[^"]*"/"OPENROUTER_API_KEY": "'"$NEW_KEY"'"/' "$file"
        echo "成功更新: $file"
    else
        echo "跳过: 未找到文件 $file"
    fi
}

# 执行替换
update_json_key "$FILE1"
update_json_key "$FILE2"
