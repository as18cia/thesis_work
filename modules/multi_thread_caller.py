import queue
import threading
from typing import Callable
from collections.abc import Iterable


class MultiThreadCaller:

    def run(self, items: list, function_to_call: Callable, number_of_threads: int,
            *args) -> list:
        in_q = queue.Queue()
        out_q = queue.Queue()
        for i, item in enumerate(items):
            in_q.put((i, item))

        threads = [threading.Thread(target=self._work, args=(in_q, out_q, function_to_call, *args)) for i in
                   range(number_of_threads)]

        # starting the threads
        for thread in threads:
            thread.start()

        # waiting for the threads to finish
        for thread in threads:
            thread.join()

        return list(out_q.queue)

    @staticmethod
    def _work(in_q, out_q, function_to_call, *args):
        while not in_q.empty():
            item = in_q.get()
            index = item[0]
            result = function_to_call(item[1], *args)
            if isinstance(result, Iterable):
                for r in result:
                    out_q.put(r)
            else:
                out_q.put(result)
            print(index)
