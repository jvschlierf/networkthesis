import os, sys
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='File to run through Pipeline')
parser.add_argument('-outfile',type=str, metavar='-o',required=False, help="Uncompress to different file name", default=None)


args = parser.parse_args(sys.argv[1:])
outfile = args.file

if args.outfile:
    outfile = args.outfile

os.chdir('../../Files/')
cl = f'zstd -d {args.file}.zst --memory=2048MB -o {outfile}.txt'
cl2 = f'bzip2 -z {outfile}.txt'
os.system(cl)
os.system(cl2)
print('Recompression done')
