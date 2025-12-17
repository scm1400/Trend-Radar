#!/bin/bash
set -e

CONFIG_DIR="/app/config"
DEFAULT_CONFIG_DIR="/app/config.defaults"

CONFIG_FILE="${CONFIG_DIR}/config.yaml"
FREQ_FILE="${CONFIG_DIR}/frequency_words.txt"

DEFAULT_CONFIG_FILE="${DEFAULT_CONFIG_DIR}/config.yaml"
DEFAULT_FREQ_FILE="${DEFAULT_CONFIG_DIR}/frequency_words.txt"

mkdir -p "${CONFIG_DIR}"

ensure_file_or_fallback() {
    local target_file="$1"
    local default_file="$2"
    local env_var="$3"

    # default file must exist in image
    if [ ! -f "${default_file}" ]; then
        echo "❌ 缺少默认配置文件: ${default_file}"
        exit 1
    fi

    # target exists -> use it
    if [ -f "${target_file}" ]; then
        export "${env_var}=${target_file}"
        return 0
    fi

    echo "⚠️ 未找到配置文件: ${target_file}"
    echo "➡️ 尝试从默认配置复制: ${default_file} -> ${target_file}"

    # try copy (may fail when /app/config is mounted read-only)
    if cp "${default_file}" "${target_file}" 2>/dev/null; then
        echo "✅ 已复制默认配置到: ${target_file}"
        export "${env_var}=${target_file}"
        return 0
    fi

    echo "⚠️ 无法写入 ${target_file}（可能是只读挂载/权限不足）。"
    echo "➡️ 将使用默认配置文件运行: ${default_file}"
    export "${env_var}=${default_file}"
}

ensure_file_or_fallback "${CONFIG_FILE}" "${DEFAULT_CONFIG_FILE}" "CONFIG_PATH"
ensure_file_or_fallback "${FREQ_FILE}" "${DEFAULT_FREQ_FILE}" "FREQUENCY_WORDS_PATH"

# 保存环境变量
env >> /etc/environment

case "${RUN_MODE:-cron}" in
"once")
    echo "🔄 单次执行"
    exec /usr/local/bin/python main.py
    ;;
"cron")
    # 生成 crontab
    echo "${CRON_SCHEDULE:-*/30 * * * *} cd /app && /usr/local/bin/python main.py" > /tmp/crontab
    
    echo "📅 生成的crontab内容:"
    cat /tmp/crontab

    if ! /usr/local/bin/supercronic -test /tmp/crontab; then
        echo "❌ crontab格式验证失败"
        exit 1
    fi

    # 立即执行一次（如果配置了）
    if [ "${IMMEDIATE_RUN:-false}" = "true" ]; then
        echo "▶️ 立即执行一次"
        /usr/local/bin/python main.py
    fi

    # 启动 Web 服务器（如果配置了）
    if [ "${ENABLE_WEBSERVER:-false}" = "true" ]; then
        echo "🌐 启动 Web 服务器..."
        /usr/local/bin/python manage.py start_webserver
    fi

    echo "⏰ 启动supercronic: ${CRON_SCHEDULE:-*/30 * * * *}"
    echo "🎯 supercronic 将作为 PID 1 运行"

    exec /usr/local/bin/supercronic -passthrough-logs /tmp/crontab
    ;;
*)
    exec "$@"
    ;;
esac