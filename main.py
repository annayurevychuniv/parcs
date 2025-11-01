from __future__ import print_function, division
from Pyro4 import expose
import random
import time
import math

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers

    @staticmethod
    def generate_random_matrix(size, value_range=(1, 10)):
        matrix = []
        for _ in range(size):
            row = [random.randint(*value_range) for _ in range(size)]
            matrix.append(row)
        return matrix

    def solve(self):
        n = self.read_input()
        matrix = self.generate_random_matrix(n)

        original_matrix = [row[:] for row in matrix]

        start_time = time.time()
        flat_matrix = []
        for row in matrix:
            flat_matrix.extend(row)

        num_workers = len(self.workers)
        elements_per_worker = len(flat_matrix) // num_workers

        mapped = []
        for i in range(num_workers):
            start_idx = i * elements_per_worker
            end_idx = start_idx + elements_per_worker if i < num_workers - 1 else len(flat_matrix)

            if start_idx >= len(flat_matrix):
                break

            chunk = flat_matrix[start_idx:end_idx]
            mapped.append(self.workers[i].compute_partial_sum(chunk))

        partial_sums = [result.value for result in mapped]
        total_sum = sum(partial_sums)
        euclidean_norm = math.sqrt(total_sum)
        total_time = time.time() - start_time

        approx_norm = self.approximate_norm(matrix, sample_size=1000)

        self.write_output(original_matrix, euclidean_norm, total_time, n, len(mapped), approx_norm)

    @staticmethod
    @expose
    def compute_partial_sum(chunk):
        return sum(x * x for x in chunk)

    @staticmethod
    def approximate_norm(matrix, sample_size=1000):
        flat = [x for row in matrix for x in row]
        k = min(sample_size, len(flat))
        sampled = random.sample(flat, k)
        sum_squares = sum(x*x for x in sampled)
        total_sum_approx = (len(flat)/k) * sum_squares
        return math.sqrt(total_sum_approx)

    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            n = int(f.readline().strip())
        return n

    def write_output(self, matrix, euclidean_norm, total_time, n, num_workers, approx_norm):
        with open(self.output_file_name, 'w') as f:
            f.write("Matrix: {} x {}\n".format(n, n))
            f.write("Workers: {}\n".format(num_workers))
            f.write("Time: {:.6f} seconds\n".format(total_time))
            f.write("Euclidean Norm: {:.6f}\n".format(euclidean_norm))
            f.write("\n")
            f.write("Approximate Norm: {:.6f}\n".format(approx_norm))
