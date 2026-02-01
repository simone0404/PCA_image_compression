import time
import functools

def time_count(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"Tempo di esecuzione di: {f.__name__!r}: {end - start:.4f}")
        return result
    return wrapper
