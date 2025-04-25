import multiprocessing as mp
import time

# Worker process function
def worker(child_conn):
    while True:
        msg = child_conn.recv()  # Wait for a message from parent
        print(f"[Worker] Got message: {msg}")

        if msg == "stop":
            child_conn.send("Worker shutting down.")
            break
        else:
            reply = f"Processed: {msg}"
            child_conn.send(reply)

# Main process
if __name__ == "__main__":
    parent_conn, child_conn = mp.Pipe()

    # Create and start worker process
    process = mp.Process(target=worker, args=(child_conn,))
    process.start()

    # Send messages to worker
    parent_conn.send("Hello Worker")
    print("[Parent] Received:", parent_conn.recv())

    parent_conn.send("Another task")
    print("[Parent] Received:", parent_conn.recv())

    parent_conn.send("stop")
    print("[Parent] Received:", parent_conn.recv())

    process.join()
    print("[Parent] Worker has shut down.")
