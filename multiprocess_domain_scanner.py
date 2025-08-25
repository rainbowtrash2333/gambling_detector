#!/usr/bin/env python3
r"""
超高性能域名扫描器 - 多进程+异步处理
扫描所有 ^www\.[a-z]{2}\d{4}\.com$ 格式的域名 (约676万个)
使用多进程+异步IO实现最大扫描效率
"""
import asyncio
import multiprocessing as mp
import socket
import csv
import os
import sys
import time
import signal
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Optional, Iterator, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import logging
from pathlib import Path
import psutil
import threading
from queue import Queue
import gc

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiProcessDomainScanner:
    def __init__(self, 
                 num_processes: int = None,
                 concurrent_dns_per_process: int = 500,
                 concurrent_port_per_process: int = 800,
                 concurrent_http_per_process: int = 200,
                 dns_timeout: int = 2,
                 port_timeout: int = 1,
                 http_timeout: int = 3,
                 batch_size: int = 5000,
                 save_interval: int = 10000):
        """
        超高性能多进程域名扫描器
        """
        self.num_processes = num_processes or mp.cpu_count() * 2
        self.concurrent_dns_per_process = concurrent_dns_per_process
        self.concurrent_port_per_process = concurrent_port_per_process
        self.concurrent_http_per_process = concurrent_http_per_process
        self.dns_timeout = dns_timeout
        self.port_timeout = port_timeout
        self.http_timeout = http_timeout
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        # 创建工作目录
        self.work_dir = Path("scan_results")
        self.work_dir.mkdir(exist_ok=True)
        
        # 结果文件
        self.results_file = self.work_dir / "domain_scan_results.csv"
        self.progress_file = self.work_dir / "scan_progress.json"
        self.temp_results_dir = self.work_dir / "temp_results"
        self.temp_results_dir.mkdir(exist_ok=True)
        
        # 统计信息
        self.global_stats = mp.Manager().dict({
            'total_domains': 676 * 10000,  # aa0000-zz9999
            'domains_processed': 0,
            'dns_resolved': 0,
            'ports_open': 0,
            'start_time': time.time(),
            'processes_active': 0
        })
        
        # 进程控制
        self.stop_event = mp.Event()
        self.result_queue = mp.Queue()
        
        # 锁
        self.stats_lock = mp.Lock()
        
    def generate_domain_ranges(self) -> List[Tuple[str, str, int, int]]:
        r"""
        生成域名范围，用于分配给不同进程
        返回: [(first_letter, second_letter, start_num, end_num), ...]
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        ranges = []
        
        # 计算每个进程处理的域名数量
        total_combinations = len(letters) * len(letters)  # 26*26 = 676
        combinations_per_process = max(1, total_combinations // self.num_processes)
        
        combinations = [(l1, l2) for l1 in letters for l2 in letters]
        
        for i in range(0, len(combinations), combinations_per_process):
            batch = combinations[i:i + combinations_per_process]
            
            for first_letter, second_letter in batch:
                # 每个字母组合包含10000个数字(0000-9999)
                ranges.append((first_letter, second_letter, 0, 9999))
        
        logger.info(f"生成了 {len(ranges)} 个域名范围，分配给 {self.num_processes} 个进程")
        return ranges
    
    def load_progress(self) -> Dict[str, Any]:
        """
        加载扫描进度
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        return {}
    
    def save_progress(self, progress_data: Dict[str, Any]):
        """
        保存扫描进度
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    async def dns_lookup_async(self, domain: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """
        异步DNS解析
        """
        async with semaphore:
            try:
                loop = asyncio.get_event_loop()
                # 去掉www前缀进行DNS查询
                clean_domain = domain.replace('www.', '')
                ip = await asyncio.wait_for(
                    loop.run_in_executor(None, socket.gethostbyname, clean_domain),
                    timeout=self.dns_timeout
                )
                return ip
            except Exception:
                return None
    
    async def check_port_async(self, ip: str, port: int, semaphore: asyncio.Semaphore) -> bool:
        """
        异步端口检查
        """
        async with semaphore:
            try:
                future = asyncio.open_connection(ip, port)
                reader, writer = await asyncio.wait_for(future, timeout=self.port_timeout)
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                return False
    
    async def check_http_status(self, url: str, semaphore: asyncio.Semaphore) -> int:
        """
        异步HTTP状态检查
        返回: 状态码
        """
        async with semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=self.http_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, allow_redirects=True) as response:
                        status = response.status
                        logger.debug(f"HTTP检查: {url} -> 状态码 {status}")
                        return status
            except Exception as e:
                logger.debug(f"HTTP检查失败: {url} - {e}")
                return 0
    
    async def process_domain_batch(self, domains: List[str], process_id: int) -> List[Tuple[str, str, int, int]]:
        """
        异步处理一批域名
        """
        results = []
        
        # 创建信号量
        dns_semaphore = asyncio.Semaphore(self.concurrent_dns_per_process)
        port_semaphore = asyncio.Semaphore(self.concurrent_port_per_process)
        http_semaphore = asyncio.Semaphore(self.concurrent_http_per_process)
        
        # DNS解析任务
        dns_tasks = [self.dns_lookup_async(domain, dns_semaphore) for domain in domains]
        dns_results = await asyncio.gather(*dns_tasks, return_exceptions=True)
        
        # 处理DNS解析成功的域名
        valid_domains = []
        for domain, ip in zip(domains, dns_results):
            if ip and not isinstance(ip, Exception):
                valid_domains.append((domain, ip))
        
        if not valid_domains:
            return results
        
        # 端口扫描任务
        port_tasks = []
        for domain, ip in valid_domains:
            for port in [80, 443]:
                port_tasks.append(self.check_port_async(ip, port, port_semaphore))
        
        port_results = await asyncio.gather(*port_tasks, return_exceptions=True)
        
        # 处理端口扫描结果
        open_services = []
        idx = 0
        for domain, ip in valid_domains:
            for port in [80, 443]:
                if idx < len(port_results) and port_results[idx] and not isinstance(port_results[idx], Exception):
                    open_services.append((domain, ip, port))
                idx += 1
        
        # HTTP检查和截图
        if open_services:
            http_tasks = []
            for domain, ip, port in open_services:
                protocol = "https" if port == 443 else "http"
                url = f"{protocol}://{domain}"
                http_tasks.append(self.check_http_status(url, http_semaphore))
            
            http_results = await asyncio.gather(*http_tasks, return_exceptions=True)
            
            # 处理HTTP检查结果并存储到results
            for (domain, ip, port), status_code in zip(open_services, http_results):
                if isinstance(status_code, int) and status_code > 0:
                    # 存储结果：(域名, IP, 端口, HTTP状态码)
                    results.append((domain, ip, port, status_code))
                    logger.debug(f"进程 {process_id}: HTTP检查完成 {domain} (HTTP: {status_code})")
                else:
                    # HTTP检查失败，状态码设为0
                    results.append((domain, ip, port, 0))
                    logger.debug(f"进程 {process_id}: HTTP检查失败 {domain}")
        
        # 更新统计
        with self.stats_lock:
            self.global_stats['domains_processed'] += len(domains)
            self.global_stats['dns_resolved'] += len(valid_domains)
            self.global_stats['ports_open'] += len(results)
        
        return results

def worker_process(process_id: int, domain_ranges: List[Tuple[str, str, int, int]], 
                  scanner_config: Dict[str, Any], result_queue: mp.Queue, 
                  stop_event: mp.Event, global_stats: Dict[str, Any], stats_lock: mp.Lock):
    """
    工作进程函数
    """
    try:
        logger.info(f"进程 {process_id} 启动，处理 {len(domain_ranges)} 个域名范围")
        
        # 创建扫描器实例
        scanner = MultiProcessDomainScanner(**scanner_config)
        scanner.global_stats = global_stats
        scanner.stats_lock = stats_lock
        
        # 更新活跃进程数
        with stats_lock:
            global_stats['processes_active'] += 1
        
        async def process_ranges():
            all_results = []
            
            for first_letter, second_letter, start_num, end_num in domain_ranges:
                if stop_event.is_set():
                    break
                
                # 生成当前范围的域名
                domains = []
                for num in range(start_num, end_num + 1):
                    if stop_event.is_set():
                        break
                    domain = f"www.{first_letter}{second_letter}{num:04d}.com"
                    domains.append(domain)
                    
                    # 批量处理
                    if len(domains) >= scanner.batch_size:
                        batch_results = await scanner.process_domain_batch(domains, process_id)
                        all_results.extend(batch_results)
                        domains = []
                        
                        # 定期保存结果
                        if len(all_results) >= scanner.save_interval:
                            result_queue.put((process_id, all_results))
                            all_results = []
                            
                        # 内存清理
                        gc.collect()
                
                # 处理剩余域名
                if domains and not stop_event.is_set():
                    batch_results = await scanner.process_domain_batch(domains, process_id)
                    all_results.extend(batch_results)
            
            # 发送最终结果
            if all_results:
                result_queue.put((process_id, all_results))
        
        # 运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(process_ranges())
        finally:
            loop.close()
            
        # 更新活跃进程数
        with stats_lock:
            global_stats['processes_active'] -= 1
            
        logger.info(f"进程 {process_id} 完成")
        
    except Exception as e:
        logger.error(f"进程 {process_id} 发生错误: {e}")
        with stats_lock:
            global_stats['processes_active'] -= 1

class ScanManager:
    """
    扫描管理器 - 负责协调多个进程和结果汇总
    """
    def __init__(self, scanner: MultiProcessDomainScanner):
        self.scanner = scanner
        self.processes = []
        self.result_writer_thread = None
        self.monitor_thread = None
        self.running = True
        
    def result_writer(self):
        """
        结果写入线程
        """
        all_results = []
        
        while self.running or not self.scanner.result_queue.empty():
            try:
                # 从队列获取结果
                process_id, results = self.scanner.result_queue.get(timeout=1)
                all_results.extend(results)
                
                logger.info(f"收到进程 {process_id} 的 {len(results)} 个结果")
                
                # 批量写入CSV文件
                if len(all_results) >= self.scanner.save_interval:
                    self.save_results_batch(all_results)
                    all_results = []
                    
            except:
                continue
        
        # 保存剩余结果
        if all_results:
            self.save_results_batch(all_results)
    
    def save_results_batch(self, results: List[Tuple[str, str, int, int]]):
        """
        批量保存结果到CSV
        数据格式: (域名, IP, 端口, HTTP状态码)
        """
        try:
            file_exists = self.scanner.results_file.exists()
            
            with open(self.scanner.results_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                if not file_exists:
                    writer.writerow(['域名', 'IP', '端口', 'HTTP状态码'])
                
                # 写入数据
                writer.writerows(results)
            
            logger.info(f"保存了 {len(results)} 个结果到 {self.scanner.results_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def monitor_progress(self):
        """
        监控进度线程
        """
        start_time = time.time()
        
        while self.running:
            try:
                time.sleep(10)  # 每10秒更新一次
                
                with self.scanner.stats_lock:
                    stats = dict(self.scanner.global_stats)
                
                elapsed = time.time() - start_time
                processed = stats['domains_processed']
                total = stats['total_domains']
                
                if processed > 0:
                    rate = processed / elapsed
                    eta = (total - processed) / rate if rate > 0 else 0
                    
                    progress = (processed / total) * 100
                    
                    print(f"\r进度: {progress:.2f}% ({processed:,}/{total:,}) | "
                          f"速度: {rate:.0f} 域名/秒 | "
                          f"DNS成功: {stats['dns_resolved']:,} | "
                          f"开放端口: {stats['ports_open']:,} | "
                          f"活跃进程: {stats['processes_active']} | "
                          f"预计剩余: {eta/3600:.1f}小时", end='', flush=True)
                
                # 保存进度
                progress_data = {
                    'processed': processed,
                    'total': total,
                    'start_time': start_time,
                    'current_time': time.time(),
                    'stats': stats
                }
                self.scanner.save_progress(progress_data)
                
            except Exception as e:
                logger.error(f"监控线程错误: {e}")
    
    def run(self):
        """
        运行扫描任务
        """
        try:
            logger.info(f"启动扫描管理器，使用 {self.scanner.num_processes} 个进程")
            
            # 生成域名范围
            domain_ranges = self.scanner.generate_domain_ranges()
            
            # 分配任务给进程
            ranges_per_process = len(domain_ranges) // self.scanner.num_processes
            
            # 启动结果写入线程
            self.result_writer_thread = threading.Thread(target=self.result_writer)
            self.result_writer_thread.start()
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self.monitor_progress)
            self.monitor_thread.start()
            
            # 启动工作进程
            for i in range(self.scanner.num_processes):
                start_idx = i * ranges_per_process
                end_idx = start_idx + ranges_per_process if i < self.scanner.num_processes - 1 else len(domain_ranges)
                process_ranges = domain_ranges[start_idx:end_idx]
                
                scanner_config = {
                    'num_processes': self.scanner.num_processes,
                    'concurrent_dns_per_process': self.scanner.concurrent_dns_per_process,
                    'concurrent_port_per_process': self.scanner.concurrent_port_per_process,
                    'concurrent_http_per_process': self.scanner.concurrent_http_per_process,
                    'dns_timeout': self.scanner.dns_timeout,
                    'port_timeout': self.scanner.port_timeout,
                    'http_timeout': self.scanner.http_timeout,
                    'batch_size': self.scanner.batch_size,
                    'save_interval': self.scanner.save_interval
                }
                
                process = mp.Process(
                    target=worker_process,
                    args=(i, process_ranges, scanner_config, self.scanner.result_queue,
                          self.scanner.stop_event, self.scanner.global_stats, self.scanner.stats_lock)
                )
                process.start()
                self.processes.append(process)
                
                logger.info(f"启动进程 {i}，处理 {len(process_ranges)} 个范围")
            
            # 等待所有进程完成
            for process in self.processes:
                process.join()
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止扫描...")
            self.stop()
        except Exception as e:
            logger.error(f"扫描管理器错误: {e}")
            self.stop()
        finally:
            self.cleanup()
    
    def stop(self):
        """
        停止扫描
        """
        self.scanner.stop_event.set()
        self.running = False
        
        # 终止所有进程
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    
    def cleanup(self):
        """
        清理资源
        """
        self.running = False
        
        # 等待线程结束
        if self.result_writer_thread:
            self.result_writer_thread.join(timeout=10)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 打印最终统计
        self.print_final_stats()
    
    def print_final_stats(self):
        """
        打印最终统计信息
        """
        with self.scanner.stats_lock:
            stats = dict(self.scanner.global_stats)
        
        elapsed = time.time() - stats['start_time']
        
        print(f"\n\n{'='*80}")
        print("扫描完成 - 最终统计报告")
        print(f"{'='*80}")
        print(f"总处理域名数: {stats['domains_processed']:,}")
        print(f"DNS解析成功: {stats['dns_resolved']:,}")
        print(f"发现开放端口: {stats['ports_open']:,}")
        print(f"总耗时: {elapsed/3600:.2f} 小时")
        print(f"平均速度: {stats['domains_processed']/elapsed:.0f} 域名/秒")
        print(f"结果文件: {self.scanner.results_file}")
        print(f"工作目录: {self.scanner.work_dir}")
        print(f"{'='*80}")

def main():
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    print("超高性能多进程域名扫描器")
    print(r"目标格式: ^www\.[a-z]{2}\d{4}\.com$")
    print(f"预计扫描数量: 6,760,000 个域名")
    print(f"CPU核心数: {mp.cpu_count()}")
    print(f"系统内存: {psutil.virtual_memory().total // (1024**3)} GB")
    print("="*60)
    
    # 创建扫描器
    scanner = MultiProcessDomainScanner(
        num_processes=mp.cpu_count() * 3,  # 进程数 = CPU核心数 * 3
        concurrent_dns_per_process=800,    # 每进程DNS并发数
        concurrent_port_per_process=1200,  # 每进程端口扫描并发数
        concurrent_http_per_process=300,   # 每进程HTTP并发数
        dns_timeout=2,                     # DNS超时
        port_timeout=1,                    # 端口超时
        http_timeout=3,                    # HTTP超时
        batch_size=3000,                   # 批处理大小
        save_interval=5000                 # 保存间隔
    )
    
    # 创建管理器
    manager = ScanManager(scanner)
    
    # 信号处理
    def signal_handler(signum, frame):
        logger.info("收到中断信号，正在优雅退出...")
        manager.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 开始扫描
        manager.run()
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()