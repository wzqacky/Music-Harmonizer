import argparse

def harmonize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', default='test.wav', type=str)
    return parser 
    