import sqlite3
import threading
import queue
import psutil
import time
import os
import numpy as np

class MetricsLogger:
    def __init__(self, device, db_path="metrics.db", flush_interval=100):
        self.device = device
        self.db_path = db_path
        self.flush_interval = flush_interval
        self.session_id = int(time.time())
        self.process = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None) # Initialize psutil cpu baseline
        
        self.queue = queue.Queue()
        self.buffer = []
        
        self.running = True
        self._init_db()
        
        self.worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.worker_thread.start()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                session_id INTEGER,
                frame_idx INTEGER,
                opencv_ms REAL,
                math_ms REAL,
                total_ms REAL,
                is_new BOOLEAN,
                cpu_percent REAL,
                memory_mb REAL,
                device TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log(self, frame_idx, opencv_ms, math_ms, total_ms, is_new):
        cpu_p = psutil.cpu_percent(interval=None)
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        
        row = (
            self.session_id, frame_idx, opencv_ms, math_ms, total_ms,
            is_new, cpu_p, mem_mb, self.device
        )
        self.buffer.append(row)
        
        if len(self.buffer) >= self.flush_interval:
            self.queue.put(self.buffer)
            self.buffer = []

    def _db_worker(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        while self.running or not self.queue.empty():
            try:
                batch = self.queue.get(timeout=0.5)
                c.executemany('''
                    INSERT INTO metrics (
                        session_id, frame_idx, opencv_ms, math_ms, total_ms,
                        is_new, cpu_percent, memory_mb, device
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                conn.commit()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Monitoring] DB Write Error: {e}")
        conn.close()

    def close(self):
        self.running = False
        if self.buffer:
            self.queue.put(self.buffer)
            self.buffer = []
        self.worker_thread.join(timeout=3.0)

    def summarize(self, exclude_warmup_frames=10):
        # Flush whatever is left just in case before summarizing
        time.sleep(0.5)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT opencv_ms, math_ms, total_ms, cpu_percent, memory_mb 
            FROM metrics 
            WHERE session_id=? AND frame_idx > ? 
            ORDER BY frame_idx
        ''', (self.session_id, exclude_warmup_frames))
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            print("[Monitoring] No data to summarize.")
            return

        data = np.array(rows)
        opencv_ms = data[:, 0]
        math_ms = data[:, 1]
        total_ms = data[:, 2]
        cpu_p = data[:, 3]
        mem_mb = data[:, 4]
        
        print("\n========================================================================================")
        print("               STEADY-STATE BENCHMARK SUMMARY (SQLITE)")
        print("========================================================================================")
        print(f"Device: {self.device}")
        print("-" * 88)
        print(f"{'Stage':<33} {'Min (ms)':>8} {'Max (ms)':>9} {'Mean (ms)':>10} {'Median':>9} {'P95 (ms)':>9} {'P99 (ms)':>9}")
        print("-" * 88)
        
        def fmt(arr):
            return f"{np.min(arr):8.2f} {np.max(arr):9.2f} {np.mean(arr):10.2f} {np.median(arr):9.2f} {np.percentile(arr, 95):9.2f} {np.percentile(arr, 99):9.2f}"
            
        print(f"{'1. OpenCV Stage':<33} {fmt(opencv_ms)}")
        print(f"{'2. Math/Inference Stage':<33} {fmt(math_ms)}")
        print(f"{'3. End-to-End Latency':<33} {fmt(total_ms)}")
        print("-" * 88)
        print("System Metrics (Per Frame):")
        print(f"   CPU Usage        : {np.mean(cpu_p):.1f}% avg, {np.max(cpu_p):.1f}% peak")
        print(f"   RAM Usage        : {np.mean(mem_mb):.1f} MB avg, {np.max(mem_mb):.1f} MB peak")
        print(f"   Frames Analyzed  : {len(rows)}")
        print("========================================================================================")
