import asyncio
import aiohttp
import time
import argparse
from statistics import mean

latencies = []

async def send_query(session, url, query, id):
    payload = {"query": query}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            await response.json()
            latency = time.perf_counter() - start
            latencies.append(latency)
    except Exception as e:
        print(f"[{id}] Error: {e}")

async def load_test(url, query, rps, duration):
    total_requests = rps * duration
    interval = 1.0 / rps

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(total_requests):
            task = send_query(session, url, query, i)
            tasks.append(task)
            await asyncio.sleep(interval)
        await asyncio.gather(*tasks)

    avg_latency = mean(latencies)
    throughput = len(latencies) / duration
    print(f"\n🧪 Test Results (rps={rps}, duration={duration}s):")
    print(f" - Total Requests Sent: {total_requests}")
    print(f" - Total Successful: {len(latencies)}")
    print(f" - Average Latency: {avg_latency:.3f} seconds")
    print(f" - Throughput: {throughput:.2f} requests/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/rag")
    parser.add_argument("--query", default="Which animals can hover in the air?")
    parser.add_argument("--rps", type=int, default=1)
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(load_test(args.url, args.query, args.rps, args.duration))

