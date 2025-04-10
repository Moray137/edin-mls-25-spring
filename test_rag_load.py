import asyncio
import aiohttp
import time
import argparse
from statistics import mean

async def send_query(session, url, query, id):
    payload = {"query": query}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            await response.json()
            latency = time.perf_counter() - start
            return latency
    except Exception as e:
        print(f"[{id}] Error: {e}")
        return None

async def run_single_test(url, query, rps, duration):
    total_requests = rps * duration
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.perf_counter()  # 请求发送开始时间
        
        for i in range(total_requests):
            target_time = start_time + i / rps
            now = time.perf_counter()
            sleep_time = target_time - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            task = asyncio.create_task(send_query(session, url, query, i))
            tasks.append(task)
        
        # 等待所有请求完成
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()  # 所有响应完成的时间

    latencies = [lat for lat in results if lat is not None]
    total_time = end_time - start_time  # 真正的测试运行时长
    if latencies:
        avg_latency = mean(latencies)
        throughput = len(latencies) / total_time
    else:
        avg_latency = float('inf')
        throughput = 0

    print(f"\n🧪 Test Results (rps={rps}, duration={duration}s):")
    print(f" - Total Requests Sent: {total_requests}")
    print(f" - Total Successful: {len(latencies)}")
    print(f" - Total Execution Time: {total_time:.2f} seconds")
    print(f" - Average Latency: {avg_latency:.3f} seconds")
    print(f" - Throughput: {throughput:.2f} requests/sec")
    return {
        "rps": rps,
        "total": total_requests,
        "success": len(latencies),
        "avg_latency": avg_latency,
        "throughput": throughput,
        "execution_time": total_time
    }


async def load_tests(url, query, rps_list, duration):
    results = []
    for rps in rps_list:
        result = await run_single_test(url, query, rps, duration)
        results.append(result)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/rag", help="RAG endpoint URL")
    parser.add_argument("--query", default="Which animals can hover in the air?", help="Query to send")
    parser.add_argument("--rps", type=int, nargs="+", default=[5, 10, 20, 30], help="List of requests per second to test")
    parser.add_argument("--duration", type=int, default=15, help="Duration (in seconds) for each test")
    args = parser.parse_args()

    asyncio.run(load_tests(args.url, args.query, args.rps, args.duration))
