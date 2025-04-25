import multiprocessing as mp
import time

# Worker process function
def worker(rank, child_conn):
    while True:
        msg = child_conn.recv()  # Wait for a message from parent
        print(f"[Worker {rank}] Got message: {msg}")

        if msg == "stop":
            child_conn.send(f"[Worker {rank}] Shutting down.")
            break
        else:
            reply = f"[Worker {rank}] Processed: {msg}"
            child_conn.send(reply)

# Main process
if __name__ == "__main__":
    num_workers = 3
    pipes = []
    processes = []

    for rank in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=worker, args=(rank, child_conn))
        process.start()
        pipes.append(parent_conn)
        processes.append(process)

    # Send messages to each worker
    for i, conn in enumerate(pipes):
        conn.send(f"Hello from parent to worker {i}")
    
    for i, conn in enumerate(pipes):
        print(f"[Parent] Received from worker {i}:", conn.recv())

    # Send stop signal
    for conn in pipes:
        conn.send("stop")

    for i, conn in enumerate(pipes):
        print(f"[Parent] Received shutdown from worker {i}:", conn.recv())

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("[Parent] All workers have shut down.")
