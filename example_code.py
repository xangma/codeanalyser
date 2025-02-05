#!/usr/bin/env python3
import concurrent.futures
import time

def foo(a, b):
    # A simple function that adds two numbers.
    c = a + b
    return c

def bar(x):
    # Multiplies x by 2.
    y = x * 2
    return y

class Baz:
    def __init__(self, value):
        # Save the given value.
        self.value = value

    def compute(self, factor):
        # Multiply the stored value by the factor.
        result = self.value * factor
        return result

def worker_task(n):
    # A function meant to run in a worker process.
    time.sleep(1)
    x = n ** 2
    return x

def main():
    # Some variables in the main function.
    a = 10
    b = 20
    s = "Hello, world!"
    
    # Call a couple of functions.
    result = foo(a, b)
    print("foo result:", result)
    
    # Create an instance of Baz and use it.
    baz = Baz(result)
    computed = baz.compute(3)
    print("Baz computed:", computed)
    
    # Use a process pool to run a task concurrently.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_task, i) for i in range(3)]
        for future in concurrent.futures.as_completed(futures):
            print("Worker result:", future.result())

if __name__ == '__main__':
    main()