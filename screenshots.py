#!/usr/bin/env python3
"""
网站截图工具
读取 scan_results/domain_scan_results.csv 文件，对有效域名进行截图
- 跳过 HTTP 4xx/5xx 状态码的网站
- 处理网站跳转，等待最终页面加载完成
- 使用多进程提高截图效率
- 截图保存到 scan_results/screenshots/ 目录
- 支持 Ctrl+C 优雅退出
"""

import csv
import os
import sys
import time
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局标志用于优雅退出
shutdown_event = threading.Event()

class ScreenshotProcessor:
    def __init__(self, 
                 max_processes: int = None,
                 page_load_timeout: int = 15,
                 implicit_wait: int = 3):
        """
        截图处理器
        """
        self.max_processes = max_processes or mp.cpu_count()
        self.page_load_timeout = page_load_timeout
        self.implicit_wait = implicit_wait
        
        # 文件路径
        self.csv_file = Path("scan_results/domain_scan_results.csv")
        self.screenshot_dir = Path("scan_results/screenshots")
        
        # 确保目录存在
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_domains': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_http_error': 0,
            'skipped_existing': 0
        }
    
    def read_csv_data(self) -> List[Tuple[str, str, int, int]]:
        """
        读取CSV文件中的域名数据
        返回: [(域名, IP, 端口, HTTP状态码), ...]
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_file}")
        
        domains_data = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    domain = row.get('域名', '').strip()
                    ip = row.get('IP', '').strip()
                    port = int(row.get('端口', 0))
                    http_code = int(row.get('HTTP状态码', 0))
                    
                    if domain and ip and port:
                        domains_data.append((domain, ip, port, http_code))
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"跳过无效行: {row} - {e}")
                    continue
        
        logger.info(f"从CSV文件读取 {len(domains_data)} 条域名记录")
        return domains_data
    
    def filter_valid_domains(self, domains_data: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, int, int]]:
        """
        过滤有效域名，跳过4xx/5xx状态码的网站
        """
        valid_domains = []
        skipped_count = 0
        
        for domain, ip, port, http_code in domains_data:
            # 跳过4xx/5xx状态码
            if 400 <= http_code < 600:
                skipped_count += 1
                logger.debug(f"跳过4xx/5xx错误网站: {domain} (HTTP: {http_code})")
                continue
            
            # 跳过HTTP检查失败的网站（状态码=0）
           # if http_code == 0:
           #     skipped_count += 1
           #     logger.debug(f"跳过HTTP检查失败的网站: {domain}")
            #    continue
                
            valid_domains.append((domain, ip, port, http_code))
        
        self.stats['skipped_http_error'] = skipped_count
        logger.info(f"过滤后有效域名: {len(valid_domains)} 个，跳过: {skipped_count} 个")
        return valid_domains
    
    def get_screenshot_filename(self, domain: str, port: int) -> str:
        """
        生成截图文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        protocol = "https" if port == 443 else "http"
        # 替换域名中的特殊字符
        safe_domain = domain.replace('www.', '').replace(':', '_').replace('/', '_')
        return f"{safe_domain}_{protocol}_{timestamp}.png"
    
    def check_existing_screenshot(self, domain: str) -> bool:
        """
        检查是否已存在截图文件
        """
        safe_domain = domain.replace('www.', '').replace(':', '_').replace('/', '_')
        pattern = f"{safe_domain}_*"
        
        existing_files = list(self.screenshot_dir.glob(pattern))
        return len(existing_files) > 0

def take_screenshot(domain_data: Tuple[str, str, int, int], 
                   screenshot_dir: str, 
                   page_load_timeout: int, 
                   implicit_wait: int,
                   skip_existing: bool = True) -> Tuple[str, bool, str]:
    """
    对单个域名进行截图
    返回: (域名, 是否成功, 消息)
    """
    domain, ip, port, http_code = domain_data
    screenshot_dir = Path(screenshot_dir)
    
    try:
        # 检查是否已存在截图
        if skip_existing:
            safe_domain = domain.replace('www.', '').replace(':', '_').replace('/', '_')
            pattern = f"{safe_domain}_*"
            existing_files = list(screenshot_dir.glob(pattern))
            if existing_files:
                return domain, False, f"截图已存在: {existing_files[0].name}"
        
        # 导入selenium (在进程内导入避免主进程依赖)
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # 构建URL
        protocol = "https" if port == 443 else "http"
        url = f"{protocol}://{domain}"
        
        # 生成截图文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_domain = domain.replace('www.', '').replace(':', '_').replace('/', '_')
        screenshot_filename = f"{safe_domain}_{protocol}_{timestamp}.png"
        screenshot_path = screenshot_dir / screenshot_filename
        
        # Chrome选项配置
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 禁用日志和通知
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-default-apps')
        
        # 性能优化
        chrome_options.add_argument('--memory-pressure-off')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        
        # 创建WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # 设置超时时间
            driver.set_page_load_timeout(page_load_timeout)
            driver.implicitly_wait(implicit_wait)
            
            logger.info(f"进程 {os.getpid()}: 开始访问 {url}")
            
            # 访问网站
            driver.get(url)
            
            # 等待页面加载完成 (处理跳转)
            WebDriverWait(driver, implicit_wait).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # 额外等待时间确保跳转和动态内容加载
            time.sleep(2)
            
            # 获取最终URL (跳转后的URL)
            final_url = driver.current_url
            if final_url != url:
                logger.info(f"检测到跳转: {url} -> {final_url}")
            
            # 滚动到页面顶部
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # 截图
            driver.save_screenshot(str(screenshot_path))
            
            # 验证截图文件
            if screenshot_path.exists() and screenshot_path.stat().st_size > 1000:
                file_size_kb = screenshot_path.stat().st_size // 1024
                logger.info(f"进程 {os.getpid()}: 截图成功 {domain} -> {screenshot_filename} ({file_size_kb}KB)")
                return domain, True, f"截图成功: {screenshot_filename}"
            else:
                logger.warning(f"进程 {os.getpid()}: 截图文件无效 {domain}")
                return domain, False, "截图文件无效或过小"
                
        finally:
            driver.quit()
            
    except Exception as e:
        error_msg = f"截图失败: {str(e)}"
        logger.error(f"进程 {os.getpid()}: {domain} - {error_msg}")
        return domain, False, error_msg

def worker_initializer():
    """工作进程初始化函数"""
    # 忽略SIGINT信号，让主进程处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    print("网站截图工具")
    print("=" * 50)
    
    try:
        # 创建处理器
        processor = ScreenshotProcessor(
            max_processes=mp.cpu_count() * 2,  # 可以适当增加进程数
            page_load_timeout=30,
            implicit_wait=3
        )
        
        print(f"配置信息:")
        print(f"  最大进程数: {processor.max_processes}")
        print(f"  页面加载超时: {processor.page_load_timeout}秒")
        print(f"  隐式等待时间: {processor.implicit_wait}秒")
        print(f"  CSV文件: {processor.csv_file}")
        print(f"  截图目录: {processor.screenshot_dir}")
        print("-" * 50)
        
        # 读取CSV数据
        print("正在读取CSV文件...")
        domains_data = processor.read_csv_data()
        processor.stats['total_domains'] = len(domains_data)
        
        # 过滤有效域名
        print("正在过滤有效域名...")
        valid_domains = processor.filter_valid_domains(domains_data)
        
        if not valid_domains:
            print("没有有效的域名需要截图！")
            return
        
        print(f"准备对 {len(valid_domains)} 个域名进行截图...")
        print("-" * 50)
        
        # 统计现有截图
        existing_screenshots = list(processor.screenshot_dir.glob("*.png"))
        print(f"现有截图文件: {len(existing_screenshots)} 个")
        
        start_time = time.time()
        
        # 使用进程池执行截图任务
        with ProcessPoolExecutor(
            max_workers=processor.max_processes,
            initializer=worker_initializer
        ) as executor:
            
            # 准备任务参数
            screenshot_tasks = []
            for domain_data in valid_domains:
                task_args = (
                    domain_data,
                    str(processor.screenshot_dir),
                    processor.page_load_timeout,
                    processor.implicit_wait,
                    True  # skip_existing
                )
                screenshot_tasks.append(task_args)
            
            # 提交所有任务
            print(f"启动 {processor.max_processes} 个进程开始截图...")
            
            try:
                # 执行任务并收集结果
                results = list(executor.map(take_screenshot, 
                    *zip(*screenshot_tasks) if screenshot_tasks else ([],[],[],[],[])))
                
                # 统计结果
                for domain, success, message in results:
                    processor.stats['processed'] += 1
                    if success:
                        processor.stats['successful'] += 1
                    else:
                        processor.stats['failed'] += 1
                        if "已存在" in message:
                            processor.stats['skipped_existing'] += 1
                
            except KeyboardInterrupt:
                print("\n收到中断信号，正在停止截图任务...")
                executor.shutdown(wait=False)
                raise
        
        elapsed_time = time.time() - start_time
        
        # 显示最终统计
        print("\n" + "=" * 60)
        print("截图任务完成 - 统计报告")
        print("=" * 60)
        print(f"总域名数: {processor.stats['total_domains']:,}")
        print(f"跳过HTTP错误: {processor.stats['skipped_http_error']:,}")
        print(f"有效域名数: {len(valid_domains):,}")
        print(f"处理域名数: {processor.stats['processed']:,}")
        print(f"截图成功: {processor.stats['successful']:,}")
        print(f"截图失败: {processor.stats['failed']:,}")
        print(f"跳过已存在: {processor.stats['skipped_existing']:,}")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        
        if processor.stats['processed'] > 0:
            success_rate = (processor.stats['successful'] / processor.stats['processed']) * 100
            avg_time = elapsed_time / processor.stats['processed']
            print(f"成功率: {success_rate:.1f}%")
            print(f"平均耗时: {avg_time:.2f} 秒/域名")
        
        # 统计最终截图文件
        final_screenshots = list(processor.screenshot_dir.glob("*.png"))
        print(f"截图目录文件总数: {len(final_screenshots)}")
        print(f"截图目录: {processor.screenshot_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保已运行域名扫描程序生成CSV文件")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()