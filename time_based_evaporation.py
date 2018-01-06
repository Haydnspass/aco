import threading
import numpy as np
import time

"""
This is still dirty draft of an idea, in which evaporation takes place not based on iterations but 
Lucas-Raphael Müller"""

# Pheromon value.
a = 5

class Evaporation(threading.Thread):
    """A god which lets the world evaporate. 
    It lives until it is ment to stop (i.e. instance.join()).
    """
    def __init__(self):
        super(Evaporation, self).__init__()
        self.stoprequest = threading.Event()

    def run(self):
        global a
        # As long as we weren't asked to stop, let evaporation happen
        while not self.stoprequest.isSet():
            a *= .99
            print('Evaporation. A is: {}'.format(a))
            time.sleep(0.01)

    def join(self, timeout=None):
        self.stoprequest.set()
        super(Evaporation, self).join(timeout)

"""Stupid ant which adds just things randomly"""
def adder(i):
    global a
    for i in range(100):
        time.sleep(np.random.randint(100)/1000)
        
        a += np.random.rand() * 100
        print('Thread {} and A is: {}'.format(i,a))


if __name__ == '__main__':
    t = [None] * 10
    for i in range(10):
        t[i] = threading.Thread(target=adder, args=(i,))
        t[i].start()
    e = Evaporation()
    e.start()
    
    for i in range(10):
        t[i].join()
    e.join()
        
    print('Completely finished.')
