"""
Recursively generate the list of analogies from a list of pairs
"""

import argparse as ap
import csv
import os.path
import sys
from argparse import RawTextHelpFormatter

class processAnalogyTextfile():

	def __init__(self):
		# read the text file, define blocks as lines starting with ':' and generate a dictionary of analogies keyed by block name
		self._file = filename
		with open(self._file, mode='r', encoding='utf-8', errors='strict') as self._input_file:
			lines = self._input_file.read()
		# make a list of all the lines in the text file
		lines_list = []
		for line in lines.splitlines():
			line.strip()
			lines_list.append(line)
		# detect blocks and create a dictionary where a key is a block name and a value is a list of lines within each block
		self.analogy_blocks={}
		for line in lines_list:
			if line.startswith(':'):
				inblock = []
				currentblock = line
				self.analogy_blocks[currentblock] = inblock
			else:
				self.analogy_blocks[currentblock].append(line)   

	def make_analogy_text(self):
		# iterate through the dictionary to generate an output string
		self.output_string=""
		for block, items in self.analogy_blocks.items():
			self.output_string += block + '\n'
			# take a list of text lines and recursively compute all possible pairs
			pairs = []
			pairs = makepairs(pairs, items)
			self.output_string += ''.join(pairs)
		# write output string to file
		with open('eval_analogy.txt', mode='w', encoding='utf-8', errors='strict') as self.output_file:
			self.output_file.write(self.output_string)

# recursive pairing function
def makepairs(pairs, lines):
	if len(lines) == 1:
		return pairs
	else:
		current = lines.pop()
		for l in lines:
			pairs.append(' '.join([current, l, '\n']))
	return(makepairs(pairs, lines))

# execute
if __name__ == "__main__":
	parser = ap.ArgumentParser(description='Preprocessor to make analogies from a list of pairs. See https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.accuracy for specifications', formatter_class=RawTextHelpFormatter)
	parser.add_argument('filename', metavar='filename', nargs='?', type=str, default=sys.stdin, help='Input file. TXT file of semantically linked pairs.')

	args = parser.parse_args()
	filename = args.filename

	processedTXT = processAnalogyTextfile()
	processedTXT.make_analogy_text()
	quit()
