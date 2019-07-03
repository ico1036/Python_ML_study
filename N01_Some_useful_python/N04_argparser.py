import argparse

## Create ArgumentParser Object
parser = argparse.ArgumentParser(description='Process some integers.')


## "integer": list consits of int data type
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')

## "accumulate: --Sum -> sum built-in function is saved in namespace,  Deafualt-> Max buil-in '' 
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

## returns namepsace args using "parser.parse_args()" method
args = parser.parse_args()

## Calculate values in "integers" using functions saved in "accumulate" 
print(args.accumulate(args.integers))
