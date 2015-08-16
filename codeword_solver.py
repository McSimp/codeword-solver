from trie_structure import Trie

class CodeBreaker:

	@staticmethod
	def solve_code(code_table, search_trie, code_alphabet):
		word_list = CodeBreaker.find_words(code_table, code_alphabet)
		word_list_length = len(word_list)

		i = 0
		while i <= word_list_length and i >= 0:

			if i == word_list_length:
				if code_alphabet.done():
					code_alphabet.solved = True
					break #YAY WE FOUND AN ANSWER
				else:
					i -= 1

			curr_word = word_list[i]
			if len(curr_word.possible_words) == 0:
				curr_word.set_possible_words(search_trie, code_alphabet)

			possible_match = curr_word.get_letter_permutation()
			if possible_match:
				code_alphabet.set_temp_letters(possible_match)
				#print(curr_word.get_curr_str(code_alphabet))
				i += 1
			else:
				curr_word.clear_possible_words(code_alphabet)
				i -= 1

		return code_alphabet

	@staticmethod
	def find_words(code_table, code_alphabet):
		word_list = []
		table_width = len(code_table[0])
		table_height = len(code_table)

		# Row Pass
		for r in range(table_height):
			new_word = CodeWord()
			for c in range(table_width):
				code_no = code_table[r][c]
				if code_no == -1:
					if len(new_word) > 1:
						CodeBreaker.add_to_ordered_list(word_list, new_word)
					new_word = CodeWord()
				else:
					new_word.add_letter(code_no, code_alphabet)
					if c == table_width - 1 and len(new_word) > 1:
						CodeBreaker.add_to_ordered_list(word_list, new_word)
						new_word = CodeWord()

		# Column Pass
		for c in range(table_width):
			new_word = CodeWord()
			for r in range(table_height):
				code_no = code_table[r][c]
				if code_no == -1:
					if len(new_word) > 1:
						CodeBreaker.add_to_ordered_list(word_list, new_word)
					new_word = CodeWord()
				else:
					new_word.add_letter(code_no, code_alphabet)
					if r == table_height - 1 and len(new_word) > 1:
						CodeBreaker.add_to_ordered_list(word_list, new_word)
						new_word = CodeWord()
		return word_list

	# Words with most fixed letters are at the top, sorted by largest word size.
	@staticmethod
	def add_to_ordered_list(word_list, code_word):
		#print(code_word)
		i = 0
		while True:
			try:
				compare_word = word_list[i]
			except:
				word_list.insert(i, code_word)
				break

			if code_word.fixed_letters > compare_word.fixed_letters:
				word_list.insert(i, code_word)
				break
			elif code_word.fixed_letters == compare_word.fixed_letters:
				if len(code_word) > len(compare_word):
					word_list.insert(i, code_word)
					break
			i += 1

class CodeWord:

	def __init__(self):
		self.code_str = []

		self.edited_elements = []
		self.possible_words = []
		self.current_word = 0

		self.possible_words_hash = dict()

		self.fixed_letters = 0

	def add_letter(self, letter_no, alphabet):
		self.code_str.append(letter_no)

		if alphabet[letter_no].fixed:
			self.fixed_letters += 1

	def set_possible_words(self, search_trie, code_alphabet):

		curr_str = self.get_curr_str(code_alphabet)
		self.edited_elements = []
		for index, char in enumerate(curr_str):
			if char == '?':
				self.edited_elements.append(index)
		if not curr_str in self.possible_words_hash.keys():
			self.possible_words_hash[curr_str] = search_trie.find_matches(curr_str)
		self.possible_words = self.possible_words_hash[curr_str]
		self.current_word = 0

	def clear_possible_words(self, code_alphabet):
		self.possible_words = []
		for edited_element in self.edited_elements:
			code_alphabet.reset_index(self.code_str[edited_element])

	def get_curr_str(self, alphabet):
		curr_str = ''
		for code_char in self.code_str:
			curr_str += str(alphabet[code_char])
		return curr_str

	# Returns the possible letters, combined with their numbers.
	def get_letter_permutation(self):
		letter_permutation = []
		try:
			possible_word = self.possible_words[self.current_word]
			for index, char in enumerate(possible_word):
				letter_permutation.append([self.code_str[index], char])
			self.current_word += 1
		except:
			letter_permutation = None
		return letter_permutation


	def __len__(self):
		return len(self.code_str)

	def __str__(self):
		return str(self.code_str)

class CodeAlphabet:
	# fixed_letters in format: [[15, 'a'], [20, 'c'], etc]
	def __init__(self, fixed_letters):
		self.code_alphabet = [CodeLetter() for i in range(27)]
		self.solved = False
		for fixed_letter in fixed_letters:
			self.code_alphabet[fixed_letter[0]].set_fixed_letter(fixed_letter[1])

	def __getitem__(self, key):
		return self.code_alphabet[key]

	def done(self):
		alphabet = list('abcdefghijklmnopqrstuvwxyz')
		for code_letter in self.code_alphabet:
			try:
				if code_letter.letter != '?':
					alphabet.remove(code_letter.letter)
			except:
				return False

		if len(alphabet) == 0:
			return True
		return False

	def reset_index(self, index):
		self.code_alphabet[index].letter = '?'

	def set_temp_letters(self, temp_letters):
		for temp_letter in temp_letters:
			self.code_alphabet[temp_letter[0]].letter = temp_letter[1]

	def __str__(self):
		out_str = ''
		for index, code_char in enumerate(self.code_alphabet):
			out_str += str(index) + ':' + str(code_char) + ', '
		return out_str

class CodeLetter:

	def __init__(self):
		self.fixed = False
		self.letter = '?'
		self.hidden = False

	def set_fixed_letter(self, letter):
		self.fixed = True
		self.letter = letter

	def __str__(self):
		return self.letter

if __name__ == "__main__":
	code_table = [[19,14,22,17,21,22,16,11,-1,17,10,15, 6,26,22],
				  [ 2,-1,20,-1, 3,-1, 2,-1, 2,-1,15,-1, 3,-1,13],
				  [15, 2,18,22,20,-1, 7, 5, 3,26,17,-1,25,15,15],
				  [24,-1, 3,-1,11,-1,17,-1,20,-1, 7,-1,25,-1, 2],
				  [10, 3,14,11,-1, 1,24, 3,20,11,-1,14,22,20, 7],
				  [-1,-1,22,-1,23,-1,-1,-1, 4,-1,-1,-1, 2,-1,13],
				  [19,15, 2, 1,22,11, 7,20,-1, 1, 3, 7,13, 7,-1],
				  [15,-1,-1,-1, 9,-1,20,-1,13,-1,20,-1,-1,-1, 7],
				  [ 2,22, 6, 6,26,-1,24,10, 7, 2,22,17,26, 1,24],
				  [12,-1,26,-1,20,-1,22,-1, 6,-1,19,-1,16,-1,10],
				  [ 7, 8,24, 2, 7,12,26, 1,24,-1, 2,22, 4,15,20],
				  [ 2,-1,-1,-1, 1,-1,14,-1,15,-1,22,-1,-1,-1,26],
				  [-1,23,10,26, 1,11,-1,22, 2,24,26, 1,24,26,16],
				  [ 1,-1,26,-1,-1,-1,14,-1,-1,-1,13,-1,10,-1,-1],
				  [16, 3,14,24,-1,20,15,26, 1, 4,-1,26, 2,26, 1],
				  [15,-1,14,-1,10,-1,24,-1,22,-1,24,-1,15,-1,14],
				  [24,15,15,-1, 7, 5, 3,22,14,-1, 7, 8, 3,13, 7],
				  [16,-1,16,-1,23,-1, 1,-1, 9,-1,20,-1,18,-1, 7],
				  [10,26,11,26,20,18,-1,23,15,15,13,16,10,26,17]]

	trie = Trie()
	trie.build_trie('wordsEn.txt')

	#alphabet = '?sruyqbexvhkmdl?cpgfnjawtzi'
	#for index, letter in enumerate(alphabet):
#		if letter != '?':
#			code_alphabet = CodeAlphabet([[index,letter]])
	#		print(letter + ' ----- '+ str(CodeBreaker.solve_code(code_table, trie, code_alphabet)))

	print str(CodeBreaker.solve_code(code_table, trie, CodeAlphabet([[16, 'c']])))
