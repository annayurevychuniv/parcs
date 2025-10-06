from future import print_function, division
from Pyro4 import expose
import time
import random
import math

class Solver:
    def init(self, workers=None, input_file=None, output_file=None):
        self.workers = workers
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def generate_matrix(size, value_range=(0, 9)):
        matrix = []
        matrix = [[random.randint(*value_range) for _ in range(size)] for _ in range(size)]
        return matrix

    def solve(self):
        n = self.read_input_size()
        matrix = self.generate_matrix(n)
        original_matrix = [row[:] for row in matrix]

        start = time.time()
        matrix_elements = [elem for row in matrix for elem in row]

        num_workers = len(self.workers)
        elem_per_worker = len(matrix_elements) // num_workers

        mapped = []
        for i in range(num_workers):
            start_idx = i * elem_per_worker
            end_idx = start_idx + elem_per_worker if i < num_workers - 1 else len(matrix_elements)
            if start_idx >= len(matrix_elements):
                break
            chunk = matrix_elements[start_idx:end_idx]
            mapped.append(self.workers[i].compute_partial_sum(chunk))

        partial_sums = [result.value for result in mapped]
        total_sum = sum(partial_sums)

        euclidean_norm = math.sqrt(total_sum)
        total_time = time.time() - start

        self.write_output(original_matrix, euclidean_norm, total_time, n, len(mapped))

    @staticmethod
    @expose
    def compute_partial_sum(chunk):
        return sum(x * x for x in chunk)

    def read_input_size(self):
        with open(self.input_file, 'r') as f:
            n = int(f.readline().strip())
        return n

    def write_output(self, matrix, euclidean_norm, total_time, size, num_workers):
        with open(self.output_file, 'w') as f:
            f.write("Matrix: {} x {}\n".format(size, size))
            f.write("Workers: {}\n".format(num_workers))
            f.write("Time: {:.6f} seconds\n".format(total_time))
            f.write("Euclidean Norm: {:.6f}".format(euclidean_norm))
