# -*- coding: utf-8 -*-
"""
P1-1f: RealMLP 0726 特征集复刻 (并行版)
4 进程并行分块构建 → 合并 parquet
"""
import sys, time, os
import multiprocessing as mp
import polars as pl

REF = r"D:\mscapital-forecasting\reference"
RAW = r"D:\mscapital-forecasting\data\raw"
SEC = r"D:\mscapital-forecasting\data\processed\0726_secondly"
OUT = r"D:\mscapital-forecasting\data\processed"
CHUNK = 40000

def worker(args):
    mode, i, n_chunks = args
    sys.path.insert(0, REF)
    import rfmf_0726data_source as src
    src.BASE_PATH = RAW
    src.PROCESSED_PATH = SEC
    src.OUTPUT_PATH = SEC
    start_id = i * CHUNK
    end_id = (i + 1) * CHUNK if i < n_chunks - 1 else None
    t0 = time.time()
    p = src.get_data(mode, start_id=start_id, end_id=end_id, return_pandas=False)
    fp = f"{OUT}/f0726_{mode}_chunk{i:02d}.parquet"
    p.write_parquet(fp)
    return i, p.shape, time.time() - t0

def main():
    t0 = time.time()
    # 1. resample (单进程, 一次)
    sys.path.insert(0, REF)
    import rfmf_0726data_source as src
    src.BASE_PATH = RAW
    src.PROCESSED_PATH = SEC
    src.OUTPUT_PATH = SEC
    print("=== 1. resample secondly ===", flush=True)
    src.main()
    print(f"resample done ({time.time()-t0:.0f}s)", flush=True)

    # 2. 并行构建 (只补缺失块)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for mode, n in [("train", 32), ("test", 17)]:
        for i in range(n):
            fp = f"{OUT}/f0726_{mode}_chunk{i:02d}.parquet"
            if not os.path.exists(fp):
                jobs.append((mode, i, n))
    print(f"missing chunks: {len(jobs)}", flush=True)

    n_proc = 2
    print(f"=== 2. parallel build ({n_proc} procs, {len(jobs)} chunks) ===", flush=True)
    with mp.Pool(n_proc) as pool:
        for i, shape, dt in pool.imap_unordered(worker, jobs):
            print(f"  chunk {i} {shape} ({dt:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    # 3. 合并
    print("=== 3. merge ===", flush=True)
    for mode, n in [("train", 32), ("test", 17)]:
        parts = [pl.read_parquet(f"{OUT}/f0726_{mode}_chunk{i:02d}.parquet") for i in range(n)]
        df = pl.concat(parts, how="vertical")
        df.write_parquet(f"{OUT}/f0726_{mode}.parquet")
        print(f"{mode}: {df.shape} saved", flush=True)
    print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
