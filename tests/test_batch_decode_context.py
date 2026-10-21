import asyncio
import threading
import unittest

from sparse_attn.baseline_runtime import (
    BatchCtx,
    batch_decode_ctx,
    batch_decode_scope,
    clear_batch_decode_ctx,
    set_batch_decode_ctx,
)


class BatchDecodeContextTests(unittest.TestCase):
    def tearDown(self):
        clear_batch_decode_ctx()

    def test_context_is_isolated_between_threads(self):
        """Concurrent requests must not overwrite each other's BatchCtx."""
        both_set = threading.Barrier(3)
        both_read = threading.Barrier(3)
        seen = {}

        def worker(name, length):
            ctx = BatchCtx(
                batch=1,
                seq_lens=[length],
                gen_lens=[0],
                alive=[True],
                pad_id=0,
            )
            set_batch_decode_ctx(ctx)
            both_set.wait(timeout=5)
            seen[name] = batch_decode_ctx()
            both_read.wait(timeout=5)
            clear_batch_decode_ctx()

        threads = [
            threading.Thread(target=worker, args=("short", 16)),
            threading.Thread(target=worker, args=("long", 64)),
        ]
        for thread in threads:
            thread.start()
        both_set.wait(timeout=5)
        both_read.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        self.assertEqual(seen["short"].seq_lens, [16])
        self.assertEqual(seen["long"].seq_lens, [64])
        self.assertIsNone(batch_decode_ctx())

    def test_scope_restores_owner_after_exception_and_nesting(self):
        outer = BatchCtx(1, [8], [0], [True], 0)
        inner = BatchCtx(1, [4], [0], [True], 0)

        with batch_decode_scope(outer):
            self.assertIs(batch_decode_ctx(), outer)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with batch_decode_scope(inner):
                    self.assertIs(batch_decode_ctx(), inner)
                    raise RuntimeError("boom")
            self.assertIs(batch_decode_ctx(), outer)

        self.assertIsNone(batch_decode_ctx())

    def test_context_is_isolated_between_async_tasks(self):
        async def run_workers():
            both_set = asyncio.Event()
            count = 0
            count_lock = asyncio.Lock()

            async def worker(length):
                nonlocal count
                ctx = BatchCtx(1, [length], [0], [True], 0)
                set_batch_decode_ctx(ctx)
                async with count_lock:
                    count += 1
                    if count == 2:
                        both_set.set()
                await asyncio.wait_for(both_set.wait(), timeout=5)
                await asyncio.sleep(0)
                seen = batch_decode_ctx()
                clear_batch_decode_ctx()
                return seen

            return await asyncio.gather(worker(24), worker(96))

        short, long = asyncio.run(run_workers())
        self.assertEqual(short.seq_lens, [24])
        self.assertEqual(long.seq_lens, [96])
        self.assertIsNone(batch_decode_ctx())


if __name__ == "__main__":
    unittest.main()
