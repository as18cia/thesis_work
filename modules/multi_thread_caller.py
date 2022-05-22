import queue
import threading
from typing import Callable


class MultiThreadCaller:

    def get_statements_for_papers(self, papers: list, function_to_call: Callable, number_of_threads: int,
                                  *args) -> list:
        in_q = queue.Queue()
        out_q = queue.Queue()
        for i, paper in enumerate(papers):
            in_q.put((i, paper))

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
            out_q.put(function_to_call(item[1], *args))
            print(index)
