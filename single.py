from __future__ import print_function, division
from Pyro4 import expose
import random
import time
import math

class Solver:
    def __init__(self, workers=None, input_file=None, output_file=None):
        self.workers = workers
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def generate_matrix(size, value_range=(0, 9)):
        return [[random.randint(*value_range) for _ in range(size)] for _ in range(size)]

    def solve(self):
        n = self.read_input()
        matrix = self.generate_matrix(n)
        start_time = time.time()
        result = self.workers[0].compute_euclidean_norm(matrix).value
        euclidean_norm = math.sqrt(result)
        total_time = time.time() - start_time
        self.write_output(matrix, euclidean_norm, total_time, n)

    @staticmethod
    @expose
    def compute_euclidean_norm(matrix):
        return sum(x*x for row in matrix for x in row)

    def read_input(self):
        with open(self.input_file, 'r') as f:
            return int(f.readline().strip())

    def write_output(self, matrix, euclidean_norm, total_time, n):
        with open(self.output_file, 'w') as f:
            f.write("Matrix: {} x {}\n".format(n, n))
            f.write("Time: {:.6f} seconds\n".format(total_time))
            f.write("Euclidean Norm: {:.6f}".format(euclidean_norm))
