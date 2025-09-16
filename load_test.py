import asyncio, time, statistics as stats, httpx

URL = "http://localhost:8000/predict?threshold=0.3"
payload = {
  "gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No",
  "OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
  "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check","MonthlyCharges":89.10,"TotalCharges":1068.20
}

N_REQ = 100
CONCURRENCY = 10

async def one_call(client, lat):
    t0 = time.perf_counter()
    r = await client.post(URL, json=payload)
    r.raise_for_status()
    lat.append((time.perf_counter()-t0)*1000)

async def main():
    lat = []
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(timeout=5.0, limits=limits) as client:
        # warm-up
        for _ in range(3):
            await client.post(URL, json=payload)

        tasks = [one_call(client, lat) for _ in range(N_REQ)]
        # exécuter par paquets pour respecter CONCURRENCY
        for i in range(0, N_REQ, CONCURRENCY):
            await asyncio.gather(*tasks[i:i+CONCURRENCY])

    lat.sort()
    print(f"Req: {N_REQ} · Concurrency: {CONCURRENCY}")
    print(f"P50: {lat[N_REQ//2]:.2f} ms | P95: {lat[int(N_REQ*0.95)-1]:.2f} ms | Mean: {stats.mean(lat):.2f} ms | Max: {max(lat):.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
