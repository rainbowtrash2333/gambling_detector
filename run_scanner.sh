#!/bin/bash

# 超高性能域名扫描器启动脚本
# 自动优化系统参数并启动扫描

echo "==================================="
echo "超高性能域名扫描器启动脚本"
echo "==================================="

# 检查Python版本
echo "检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
pip3 show aiohttp > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "安装依赖包..."
    pip3 install -r multiprocess_requirements.txt
fi

# 检查Chrome和ChromeDriver
echo "检查Chrome浏览器..."
which google-chrome > /dev/null 2>&1 || which chromium-browser > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "警告: 未找到Chrome浏览器，截图功能可能无法使用"
fi

echo "检查ChromeDriver..."
which chromedriver > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "警告: 未找到ChromeDriver，截图功能可能无法使用"
fi

# 系统优化
echo "优化系统参数..."

# 增加文件描述符限制
ulimit -n 65536
echo "文件描述符限制: $(ulimit -n)"

# 增加进程限制
ulimit -u 32768
echo "进程限制: $(ulimit -u)"

# 显示系统信息
echo "系统信息:"
echo "  CPU核心数: $(nproc)"
echo "  内存大小: $(free -h | grep Mem | awk '{print $2}')"
echo "  磁盘空间: $(df -h . | tail -1 | awk '{print $4}') 可用"

# 创建工作目录
mkdir -p scan_results/screenshots scan_results/temp_results

echo "==================================="
echo "启动扫描器..."
echo "按 Ctrl+C 可随时停止扫描"
echo "==================================="

# 启动扫描器
python3 multiprocess_domain_scanner.py

echo "扫描完成!"
echo "结果文件: scan_results/domain_scan_results.csv"
echo "截图目录: scan_results/screenshots/"