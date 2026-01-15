#!/usr/bin/env python3
"""
Performance benchmark for CMS and TOPK Redis commands.

Usage:
    python run_benchmark.py --host localhost --port 6379
    python run_benchmark.py --host1 server1 --port1 6379 --host2 server2 --port2 6379
"""

import argparse
import json
import random
import string
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import redis


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Stores latency measurements and computes statistics."""
    command: str
    scenario: str
    latencies_ns: List[int] = field(default_factory=list)
    errors: int = 0

    def add_measurement(self, latency_ns: int) -> None:
        self.latencies_ns.append(latency_ns)

    def record_error(self) -> None:
        self.errors += 1

    @property
    def total_ops(self) -> int:
        return len(self.latencies_ns) + self.errors

    @property
    def successful_ops(self) -> int:
        return len(self.latencies_ns)

    @property
    def throughput_ops_sec(self) -> float:
        if not self.latencies_ns:
            return 0.0
        total_time_sec = sum(self.latencies_ns) / 1e9
        return len(self.latencies_ns) / total_time_sec if total_time_sec > 0 else 0.0

    @property
    def latency_avg_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        return statistics.mean(self.latencies_ns) / 1e6

    @property
    def latency_min_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        return min(self.latencies_ns) / 1e6

    @property
    def latency_max_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        return max(self.latencies_ns) / 1e6

    @property
    def latency_p50_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        return statistics.median(self.latencies_ns) / 1e6

    @property
    def latency_p95_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        sorted_lat = sorted(self.latencies_ns)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)] / 1e6

    @property
    def latency_p99_ms(self) -> float:
        if not self.latencies_ns:
            return 0.0
        sorted_lat = sorted(self.latencies_ns)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)] / 1e6

    def to_dict(self) -> dict:
        return {
            'command': self.command,
            'scenario': self.scenario,
            'total_ops': self.total_ops,
            'successful_ops': self.successful_ops,
            'errors': self.errors,
            'throughput_ops_sec': round(self.throughput_ops_sec, 2),
            'latency_ms': {
                'avg': round(self.latency_avg_ms, 4),
                'min': round(self.latency_min_ms, 4),
                'max': round(self.latency_max_ms, 4),
                'p50': round(self.latency_p50_ms, 4),
                'p95': round(self.latency_p95_ms, 4),
                'p99': round(self.latency_p99_ms, 4),
            }
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Executes benchmarks against a Redis server."""

    def __init__(self, host: str, port: int, iterations: int = 10000,
                 warmup: int = 1000, key_prefix: str = 'bench:'):
        self.host = host
        self.port = port
        self.iterations = iterations
        self.warmup = warmup
        self.key_prefix = key_prefix
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.keys_to_cleanup: List[str] = []

    def connect(self) -> bool:
        try:
            self.client.ping()
            return True
        except redis.ConnectionError:
            return False

    def get_server_info(self) -> Dict:
        try:
            info = self.client.info()
            return {
                'redis_version': info.get('redis_version', 'unknown'),
            }
        except Exception:
            return {'redis_version': 'unknown'}

    def random_key(self, prefix: str = '') -> str:
        suffix = ''.join(random.choices(string.ascii_lowercase, k=8))
        key = f"{self.key_prefix}{prefix}:{suffix}"
        self.keys_to_cleanup.append(key)
        return key

    def cleanup(self) -> int:
        if self.keys_to_cleanup:
            try:
                deleted = self.client.delete(*self.keys_to_cleanup)
                self.keys_to_cleanup.clear()
                return deleted
            except Exception:
                self.keys_to_cleanup.clear()
                return 0
        return 0

    def run_benchmark(self, name: str, scenario: str,
                      setup: Callable, operation: Callable,
                      teardown: Optional[Callable] = None) -> BenchmarkMetrics:
        metrics = BenchmarkMetrics(command=name, scenario=scenario)

        # Setup
        ctx = setup()

        # Warmup phase
        for _ in range(self.warmup):
            try:
                operation(ctx)
            except Exception:
                pass

        # Measurement phase
        for _ in range(self.iterations):
            start = time.perf_counter_ns()
            try:
                operation(ctx)
                latency = time.perf_counter_ns() - start
                metrics.add_measurement(latency)
            except Exception:
                metrics.record_error()

        # Teardown
        if teardown:
            teardown(ctx)

        return metrics


# =============================================================================
# CMS Benchmarks
# =============================================================================

class CMSBenchmarks:
    """Benchmark implementations for CMS commands."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.client = runner.client

    def benchmark_initbydim(self, width: int, depth: int) -> BenchmarkMetrics:
        counter = [0]

        def setup():
            return {'width': width, 'depth': depth}

        def operation(ctx):
            key = f"{self.runner.key_prefix}initbydim:{counter[0]}"
            self.client.execute_command("CMS.INITBYDIM", key, ctx['width'], ctx['depth'])
            self.runner.keys_to_cleanup.append(key)
            counter[0] += 1

        return self.runner.run_benchmark(
            name='CMS.INITBYDIM',
            scenario=f'{width}x{depth}',
            setup=setup,
            operation=operation
        )

    def benchmark_initbyprob(self, error: float, probability: float) -> BenchmarkMetrics:
        counter = [0]

        def setup():
            return {'error': error, 'probability': probability}

        def operation(ctx):
            key = f"{self.runner.key_prefix}initbyprob:{counter[0]}"
            self.client.execute_command("CMS.INITBYPROB", key, ctx['error'], ctx['probability'])
            self.runner.keys_to_cleanup.append(key)
            counter[0] += 1

        return self.runner.run_benchmark(
            name='CMS.INITBYPROB',
            scenario=f'err={error},prob={probability}',
            setup=setup,
            operation=operation
        )

    def benchmark_incrby(self, width: int, depth: int, batch_size: int) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('cms_incrby')
            self.client.execute_command("CMS.INITBYDIM", key, width, depth)
            return {'key': key, 'counter': [0]}

        def operation(ctx):
            args = []
            base = ctx['counter'][0] * batch_size
            for i in range(batch_size):
                args.extend([f"item:{base + i}", 1])
            self.client.execute_command("CMS.INCRBY", ctx['key'], *args)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='CMS.INCRBY',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_query(self, width: int, depth: int, batch_size: int,
                        pre_populate: int = 1000) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('cms_query')
            self.client.execute_command("CMS.INITBYDIM", key, width, depth)
            # Pre-populate
            for i in range(0, pre_populate, 100):
                args = []
                for j in range(i, min(i + 100, pre_populate)):
                    args.extend([f"item:{j}", 1])
                self.client.execute_command("CMS.INCRBY", key, *args)
            return {'key': key, 'max_item': pre_populate, 'counter': [0]}

        def operation(ctx):
            base = (ctx['counter'][0] * batch_size) % ctx['max_item']
            items = [f"item:{(base + i) % ctx['max_item']}" for i in range(batch_size)]
            self.client.execute_command("CMS.QUERY", ctx['key'], *items)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='CMS.QUERY',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_merge(self, width: int, depth: int, num_sources: int,
                        weighted: bool = False) -> BenchmarkMetrics:
        def setup():
            sources = []
            for i in range(num_sources):
                src_key = self.runner.random_key(f'cms_src{i}')
                self.client.execute_command("CMS.INITBYDIM", src_key, width, depth)
                self.client.execute_command("CMS.INCRBY", src_key, f"item_{i}", 100)
                sources.append(src_key)
            return {'sources': sources, 'width': width, 'depth': depth,
                    'weighted': weighted, 'counter': [0]}

        def operation(ctx):
            dest_key = self.runner.random_key(f'cms_dest{ctx["counter"][0]}')
            self.client.execute_command("CMS.INITBYDIM", dest_key, ctx['width'], ctx['depth'])
            args = [dest_key, len(ctx['sources'])] + ctx['sources']
            if ctx['weighted']:
                args.append('WEIGHTS')
                args.extend([1] * len(ctx['sources']))
            self.client.execute_command("CMS.MERGE", *args)
            ctx['counter'][0] += 1

        scenario = f'{num_sources}_sources'
        if weighted:
            scenario += '_weighted'
        return self.runner.run_benchmark(
            name='CMS.MERGE',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_info(self, width: int, depth: int) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('cms_info')
            self.client.execute_command("CMS.INITBYDIM", key, width, depth)
            for i in range(0, 1000, 100):
                args = []
                for j in range(i, i + 100):
                    args.extend([f"item:{j}", 1])
                self.client.execute_command("CMS.INCRBY", key, *args)
            return {'key': key}

        def operation(ctx):
            self.client.execute_command("CMS.INFO", ctx['key'])

        return self.runner.run_benchmark(
            name='CMS.INFO',
            scenario='populated',
            setup=setup,
            operation=operation
        )

    def run_all(self, verbose: bool = False) -> List[BenchmarkMetrics]:
        results = []
        benchmarks = [
            ('CMS.INITBYDIM small', lambda: self.benchmark_initbydim(100, 5)),
            ('CMS.INITBYDIM medium', lambda: self.benchmark_initbydim(1000, 7)),
            ('CMS.INITBYDIM large', lambda: self.benchmark_initbydim(10000, 10)),
            ('CMS.INITBYPROB', lambda: self.benchmark_initbyprob(0.01, 0.01)),
            ('CMS.INCRBY single', lambda: self.benchmark_incrby(1000, 7, 1)),
            ('CMS.INCRBY batch_10', lambda: self.benchmark_incrby(1000, 7, 10)),
            ('CMS.INCRBY batch_100', lambda: self.benchmark_incrby(1000, 7, 100)),
            ('CMS.QUERY single', lambda: self.benchmark_query(1000, 7, 1)),
            ('CMS.QUERY batch_10', lambda: self.benchmark_query(1000, 7, 10)),
            ('CMS.QUERY batch_100', lambda: self.benchmark_query(1000, 7, 100)),
            ('CMS.MERGE 2 sources', lambda: self.benchmark_merge(1000, 7, 2)),
            ('CMS.MERGE weighted', lambda: self.benchmark_merge(1000, 7, 2, weighted=True)),
            ('CMS.INFO', lambda: self.benchmark_info(1000, 7)),
        ]

        for name, bench_fn in benchmarks:
            if verbose:
                print(f"  Running {name}...", end=' ', flush=True)
            result = bench_fn()
            results.append(result)
            if verbose:
                print(f"{result.throughput_ops_sec:.2f} ops/sec")
            self.runner.cleanup()

        return results


# =============================================================================
# TOPK Benchmarks
# =============================================================================

class TOPKBenchmarks:
    """Benchmark implementations for TOPK commands."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.client = runner.client

    def benchmark_reserve(self, k: int, width: int, depth: int,
                          decay: float) -> BenchmarkMetrics:
        counter = [0]

        def setup():
            return {'k': k, 'width': width, 'depth': depth, 'decay': decay}

        def operation(ctx):
            key = f"{self.runner.key_prefix}reserve:{counter[0]}"
            self.client.execute_command("TOPK.RESERVE", key,
                                        ctx['k'], ctx['width'], ctx['depth'], ctx['decay'])
            self.runner.keys_to_cleanup.append(key)
            counter[0] += 1

        return self.runner.run_benchmark(
            name='TOPK.RESERVE',
            scenario=f'k={k}',
            setup=setup,
            operation=operation
        )

    def benchmark_add(self, k: int, width: int, depth: int,
                      decay: float, batch_size: int) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_add')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            return {'key': key, 'counter': [0]}

        def operation(ctx):
            base = ctx['counter'][0] * batch_size
            items = [f"item:{base + i}" for i in range(batch_size)]
            self.client.execute_command("TOPK.ADD", ctx['key'], *items)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='TOPK.ADD',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_incrby(self, k: int, width: int, depth: int,
                         decay: float, batch_size: int) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_incrby')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            return {'key': key, 'counter': [0]}

        def operation(ctx):
            args = []
            base = ctx['counter'][0] * batch_size
            for i in range(batch_size):
                args.extend([f"item:{base + i}", 1])
            self.client.execute_command("TOPK.INCRBY", ctx['key'], *args)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='TOPK.INCRBY',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_query(self, k: int, width: int, depth: int,
                        decay: float, batch_size: int,
                        pre_populate: int = 1000) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_query')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            for i in range(pre_populate):
                self.client.execute_command("TOPK.ADD", key, f"item:{i}")
            return {'key': key, 'max_item': pre_populate, 'counter': [0]}

        def operation(ctx):
            base = (ctx['counter'][0] * batch_size) % ctx['max_item']
            items = [f"item:{(base + i) % ctx['max_item']}" for i in range(batch_size)]
            self.client.execute_command("TOPK.QUERY", ctx['key'], *items)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='TOPK.QUERY',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_count(self, k: int, width: int, depth: int,
                        decay: float, batch_size: int,
                        pre_populate: int = 1000) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_count')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            for i in range(pre_populate):
                freq = pre_populate - i
                self.client.execute_command("TOPK.INCRBY", key, f"item:{i}", freq)
            return {'key': key, 'max_item': pre_populate, 'counter': [0]}

        def operation(ctx):
            base = (ctx['counter'][0] * batch_size) % ctx['max_item']
            items = [f"item:{(base + i) % ctx['max_item']}" for i in range(batch_size)]
            self.client.execute_command("TOPK.COUNT", ctx['key'], *items)
            ctx['counter'][0] += 1

        scenario = 'single' if batch_size == 1 else f'batch_{batch_size}'
        return self.runner.run_benchmark(
            name='TOPK.COUNT',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_list(self, k: int, width: int, depth: int,
                       decay: float, withcount: bool = False,
                       pre_populate: int = 500) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_list')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            for i in range(pre_populate):
                self.client.execute_command("TOPK.ADD", key, f"item:{i}")
            return {'key': key, 'withcount': withcount}

        def operation(ctx):
            if ctx['withcount']:
                self.client.execute_command("TOPK.LIST", ctx['key'], "WITHCOUNT")
            else:
                self.client.execute_command("TOPK.LIST", ctx['key'])

        scenario = 'withcount' if withcount else 'basic'
        return self.runner.run_benchmark(
            name='TOPK.LIST',
            scenario=scenario,
            setup=setup,
            operation=operation
        )

    def benchmark_info(self, k: int, width: int, depth: int,
                       decay: float) -> BenchmarkMetrics:
        def setup():
            key = self.runner.random_key('topk_info')
            self.client.execute_command("TOPK.RESERVE", key, k, width, depth, decay)
            for i in range(100):
                self.client.execute_command("TOPK.ADD", key, f"item:{i}")
            return {'key': key}

        def operation(ctx):
            self.client.execute_command("TOPK.INFO", ctx['key'])

        return self.runner.run_benchmark(
            name='TOPK.INFO',
            scenario=f'k={k}',
            setup=setup,
            operation=operation
        )

    def run_all(self, verbose: bool = False) -> List[BenchmarkMetrics]:
        results = []
        k, width, depth, decay = 100, 500, 7, 0.9

        benchmarks = [
            ('TOPK.RESERVE k=10', lambda: self.benchmark_reserve(10, 500, 7, 0.9)),
            ('TOPK.RESERVE k=100', lambda: self.benchmark_reserve(100, 500, 7, 0.9)),
            ('TOPK.RESERVE k=1000', lambda: self.benchmark_reserve(1000, 500, 7, 0.9)),
            ('TOPK.ADD single', lambda: self.benchmark_add(k, width, depth, decay, 1)),
            ('TOPK.ADD batch_10', lambda: self.benchmark_add(k, width, depth, decay, 10)),
            ('TOPK.ADD batch_100', lambda: self.benchmark_add(k, width, depth, decay, 100)),
            ('TOPK.INCRBY single', lambda: self.benchmark_incrby(k, width, depth, decay, 1)),
            ('TOPK.INCRBY batch_10', lambda: self.benchmark_incrby(k, width, depth, decay, 10)),
            ('TOPK.INCRBY batch_100', lambda: self.benchmark_incrby(k, width, depth, decay, 100)),
            ('TOPK.QUERY single', lambda: self.benchmark_query(k, width, depth, decay, 1)),
            ('TOPK.QUERY batch_10', lambda: self.benchmark_query(k, width, depth, decay, 10)),
            ('TOPK.QUERY batch_100', lambda: self.benchmark_query(k, width, depth, decay, 100)),
            ('TOPK.COUNT single', lambda: self.benchmark_count(k, width, depth, decay, 1)),
            ('TOPK.COUNT batch_10', lambda: self.benchmark_count(k, width, depth, decay, 10)),
            ('TOPK.LIST basic', lambda: self.benchmark_list(k, width, depth, decay, withcount=False)),
            ('TOPK.LIST withcount', lambda: self.benchmark_list(k, width, depth, decay, withcount=True)),
            ('TOPK.INFO', lambda: self.benchmark_info(k, width, depth, decay)),
        ]

        for name, bench_fn in benchmarks:
            if verbose:
                print(f"  Running {name}...", end=' ', flush=True)
            result = bench_fn()
            results.append(result)
            if verbose:
                print(f"{result.throughput_ops_sec:.2f} ops/sec")
            self.runner.cleanup()

        return results


# =============================================================================
# Output Formatting
# =============================================================================

def format_table(results: dict) -> str:
    """Format results as ASCII table."""
    lines = []

    lines.append("=" * 95)
    lines.append("                      CMS/TOPK Performance Benchmark Results")
    lines.append("=" * 95)

    meta = results['metadata']
    server = meta['server']
    config = meta['config']

    lines.append(f"Server: {server['host']}:{server['port']} ({server.get('redis_version', 'unknown')})")
    lines.append(f"Iterations: {config['iterations']} | Warmup: {config['warmup']} | Date: {meta['timestamp'][:19]}")
    lines.append("")

    for category in ['CMS', 'TOPK']:
        if category not in results['results'] or not results['results'][category]:
            continue

        lines.append("-" * 95)
        lines.append(f"{category} Commands")
        lines.append("-" * 95)

        header = f"{'Command':<16} {'Scenario':<18} {'Ops/sec':>12} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10}"
        lines.append(header)
        lines.append("-" * 95)

        for r in results['results'][category]:
            lat = r['latency_ms']
            line = f"{r['command']:<16} {r['scenario'][:18]:<18} {r['throughput_ops_sec']:>12.2f} {lat['avg']:>10.4f} {lat['p50']:>10.4f} {lat['p95']:>10.4f} {lat['p99']:>10.4f}"
            lines.append(line)

        lines.append("")

    lines.append("=" * 95)
    return "\n".join(lines)


def format_comparison_table(results1: dict, results2: dict) -> str:
    """Format comparison of two server results."""
    lines = []

    lines.append("=" * 105)
    lines.append("                           Server Comparison Results")
    lines.append("=" * 105)

    s1 = results1['metadata']['server']
    s2 = results2['metadata']['server']
    lines.append(f"Server 1: {s1['host']}:{s1['port']} | Server 2: {s2['host']}:{s2['port']}")
    lines.append("")

    # Build lookup for results2
    r2_lookup = {}
    for category in ['CMS', 'TOPK']:
        if category in results2['results']:
            for r in results2['results'][category]:
                key = (r['command'], r['scenario'])
                r2_lookup[key] = r

    # Throughput comparison
    lines.append("-" * 105)
    lines.append("                                Throughput (ops/sec)")
    lines.append("-" * 105)
    header = f"{'Command':<16} {'Scenario':<18} {'Server1':>12} {'Server2':>12} {'Diff':>14} {'Ratio':>10}"
    lines.append(header)
    lines.append("-" * 105)

    for category in ['CMS', 'TOPK']:
        if category not in results1['results']:
            continue
        for r1 in results1['results'][category]:
            key = (r1['command'], r1['scenario'])
            r2 = r2_lookup.get(key)
            if r2:
                t1, t2 = r1['throughput_ops_sec'], r2['throughput_ops_sec']
                diff = t2 - t1
                ratio = t2 / t1 if t1 > 0 else 0
                diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
                line = f"{r1['command']:<16} {r1['scenario'][:18]:<18} {t1:>12.2f} {t2:>12.2f} {diff_str:>14} {ratio:>9.2f}x"
                lines.append(line)

    lines.append("")

    # Latency comparison (P99)
    lines.append("-" * 105)
    lines.append("                                P99 Latency (ms)")
    lines.append("-" * 105)
    header = f"{'Command':<16} {'Scenario':<18} {'Server1':>12} {'Server2':>12} {'Diff':>14} {'Ratio':>10}"
    lines.append(header)
    lines.append("-" * 105)

    for category in ['CMS', 'TOPK']:
        if category not in results1['results']:
            continue
        for r1 in results1['results'][category]:
            key = (r1['command'], r1['scenario'])
            r2 = r2_lookup.get(key)
            if r2:
                l1, l2 = r1['latency_ms']['p99'], r2['latency_ms']['p99']
                diff = l2 - l1
                ratio = l2 / l1 if l1 > 0 else 0
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                line = f"{r1['command']:<16} {r1['scenario'][:18]:<18} {l1:>12.4f} {l2:>12.4f} {diff_str:>14} {ratio:>9.2f}x"
                lines.append(line)

    lines.append("=" * 105)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def run_benchmarks(host: str, port: int, iterations: int, warmup: int,
                   verbose: bool = False) -> dict:
    """Run all benchmarks against a single server."""
    runner = BenchmarkRunner(
        host=host,
        port=port,
        iterations=iterations,
        warmup=warmup
    )

    if not runner.connect():
        print(f"Error: Cannot connect to {host}:{port}")
        sys.exit(1)

    server_info = runner.get_server_info()
    print(f"Connected to {host}:{port} ({server_info.get('redis_version', 'unknown')})")

    all_results = {
        'metadata': {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'server': {'host': host, 'port': port, **server_info},
            'config': {
                'iterations': iterations,
                'warmup': warmup,
            }
        },
        'results': {'CMS': [], 'TOPK': []}
    }

    # Run CMS benchmarks
    print("\nRunning CMS benchmarks...")
    cms_benchmarks = CMSBenchmarks(runner)
    cms_results = cms_benchmarks.run_all(verbose=verbose)
    all_results['results']['CMS'] = [r.to_dict() for r in cms_results]

    # Run TOPK benchmarks
    print("\nRunning TOPK benchmarks...")
    topk_benchmarks = TOPKBenchmarks(runner)
    topk_results = topk_benchmarks.run_all(verbose=verbose)
    all_results['results']['TOPK'] = [r.to_dict() for r in topk_results]

    # Final cleanup
    runner.cleanup()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CMS and TOPK Redis commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --host localhost --port 6379
  python run_benchmark.py --host1 localhost --port1 6379 --host2 10.90.0.238 --port2 6379
  python run_benchmark.py --host localhost --port 6379 --output json --output-file results.json
        """
    )

    # Server configuration (single mode)
    parser.add_argument('--host', type=str, default='localhost',
                        help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379,
                        help='Redis port (default: 6379)')

    # Server configuration (comparison mode)
    parser.add_argument('--host1', type=str,
                        help='First Redis host (comparison mode)')
    parser.add_argument('--port1', type=int,
                        help='First Redis port (comparison mode)')
    parser.add_argument('--host2', type=str,
                        help='Second Redis host (comparison mode)')
    parser.add_argument('--port2', type=int,
                        help='Second Redis port (comparison mode)')

    # Benchmark configuration
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations per benchmark (default: 10000)')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup iterations before measurement (default: 1000)')

    # Output configuration
    parser.add_argument('--output', type=str, default='table',
                        choices=['json', 'table', 'both'],
                        help='Output format (default: table)')
    parser.add_argument('--output-file', type=str,
                        help='Save JSON results to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show progress for each benchmark')

    args = parser.parse_args()

    # Determine if comparison mode
    comparison_mode = all([args.host1, args.port1, args.host2, args.port2])

    if comparison_mode:
        print("Running in comparison mode...\n")
        print(f"=== Server 1: {args.host1}:{args.port1} ===")
        results1 = run_benchmarks(args.host1, args.port1, args.iterations,
                                  args.warmup, args.verbose)
        print(f"\n=== Server 2: {args.host2}:{args.port2} ===")
        results2 = run_benchmarks(args.host2, args.port2, args.iterations,
                                  args.warmup, args.verbose)

        # Output comparison
        if args.output in ('table', 'both'):
            print("\n" + format_comparison_table(results1, results2))

        if args.output in ('json', 'both'):
            comparison = {
                'server1': results1,
                'server2': results2,
                'comparison_timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            json_output = json.dumps(comparison, indent=2)
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(json_output)
                print(f"\nResults saved to {args.output_file}")
            elif args.output == 'json':
                print(json_output)
    else:
        results = run_benchmarks(args.host, args.port, args.iterations,
                                 args.warmup, args.verbose)

        # Output results
        if args.output in ('table', 'both'):
            print("\n" + format_table(results))

        if args.output in ('json', 'both'):
            json_output = json.dumps(results, indent=2)
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(json_output)
                print(f"\nResults saved to {args.output_file}")
            elif args.output == 'json':
                print(json_output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
