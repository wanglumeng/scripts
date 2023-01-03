#!/usr/bin/env python
import os

if __name__ == '__main__':
    print(f'os.getcwd: {os.getcwd()}')
    print(f'os.path.abspath: {os.path.abspath(".")}')
    print(f'os.path.dirname(__file__): {os.path.dirname(__file__)}')
