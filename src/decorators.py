import os
import time
import functools

def timer(f):
    """Function decorator to measure the execution time of a function"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        show_time = kwargs.pop('show_time', False)
        if not show_time:
            return f(*args, **kwargs)
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"Tempo di esecuzione di {f.__name__}: {t1 - t0:.4f} secondi")
        return result
    return wrapper


def ensure_folder(f):
    """Function decorator to ensure that the output folder exists"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        folder = kwargs.get('output_folder', 'output')
        os.makedirs(folder, exist_ok=True)
        return f(*args, **kwargs)
    return wrapper
