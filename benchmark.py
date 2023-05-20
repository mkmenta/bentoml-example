import asyncio
from collections import defaultdict
import json
import time
import httpx
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import matplotlib.pyplot as plt


async def perform_request(semaphore, request):
    async with semaphore:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post("http://localhost:3000/detect",
                                             data=json.dumps(request),
                                             timeout=600,
                                             headers={
                                                 "Content-Type": "application/json",
                                             })
                if response.status_code != 200:
                    return (None, response.status_code, response.elapsed.total_seconds())
                return (response.text, response.status_code, response.elapsed.total_seconds())
            except Exception as e:
                print(e.__class__.__name__, e)
                return (None, -1, None)


async def main():
    url = "https://live.staticflickr.com/7169/6396112547_9674218be1_b_d.jpg"
    request = {
        "image_url": url
    }

    semaphore = asyncio.Semaphore(100)
    tasks = []

    # Warmup
    for _ in range(1500):
        task = asyncio.create_task(perform_request(semaphore, request))
        tasks.append(task)

    await tqdm_asyncio.gather(*tasks)

    # Benchmark
    start_t = time.time()
    tasks = []
    for _ in range(4500):
        task = asyncio.create_task(perform_request(semaphore, request))
        tasks.append(task)

    responses = await tqdm_asyncio.gather(*tasks)
    end_t = time.time()

    print(f"Total time (s);{end_t-start_t:.0f}")
    print(f"Requests per second;{len(responses)/(end_t-start_t):.0f}")
    print(f"Milliseconds per request;{(end_t-start_t)/len(responses)*1000:.0f}")

    # Count response codes
    all_responses = len(responses)
    response_codes = [r[1] for r in responses]
    response_codes = np.array(response_codes)
    print("Response codes:")
    for code, count in zip(*np.unique(response_codes, return_counts=True)):
        print(f"{code};{count};{count/all_responses*100:.2f}%")

    # Count stats
    times = defaultdict(list)

    # Real milliseconds per request
    times['elapsed_200'] = [response[2]*1000 for response in responses if response[1] == 200]
    aux = [response[2]*1000 for response in responses if response[1] != 200]
    if aux:
        times['elapsed_error'] = aux
    responses = [json.loads(response[0]) for response in responses if response[1] == 200]
    for response in responses:
        for key, seconds in response["timings"].items():
            times[key].append(seconds * 1000)
    times = {k: np.array(v) for k, v in times.items()}

    # Print stats
    print("Key;Mean;Std;Median;Min;Max")
    for key, seconds in times.items():
        print(f"{key};{np.mean(seconds):.2f}"
              f";{np.std(seconds):.2f}"
              f";{np.median(seconds):.2f}"
              f";{np.min(seconds):.2f}"
              f";{np.max(seconds):.2f}")
        plt.hist(seconds, bins=100)
        plt.title(f"Histogram of {key} (millis)")
        plt.savefig(f"{key}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    asyncio.run(main())
