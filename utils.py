import numpy as np

#==================================================================
def PrintTime(tic, toc, msg):
    """
    Print the elapsed time for a given operation.
    """
    elapsed = toc - tic
    if elapsed < 1:
        print(f"Time[{msg}] : {elapsed * 1000:.2f} ms")
    elif elapsed > 300:
        print(f"Time[{msg}] : {elapsed / 60:.2f} m")
    else:
        print(f"Time[{msg}] : {elapsed:.2f} s")