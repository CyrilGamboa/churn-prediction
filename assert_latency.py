import httpx, time
URL = "http://localhost:8000/health"
t0 = time.perf_counter()
r = httpx.get(URL, timeout=3.0)
lat_ms = (time.perf_counter()-t0)*1000
assert r.status_code == 200 and r.json().get("status")=="ok", "health failed"
assert lat_ms <= 100, f"Latency too high: {lat_ms:.2f} ms"
print(f"Health OK in {lat_ms:.2f} ms")
