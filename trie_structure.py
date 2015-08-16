__author__ = 'Robert'

class Node:
    """
    Creates basic node structure
    """

    def __init__(self, letter=None, previous=None):
        self.letter = letter
        self.next = []
        self.previous = previous
        self.isWord = False

    def __str__(self):
        return self.letter

    def letter(self):
        return self.letter


class Trie:
    """
    Creates a basic Trie Structure
    """

    def __init__(self, letter=None):
        if letter is None:
            self.root = Node('')
        else:
            self.root = Node(letter)
        self.columns = []

    def build_trie(self, file_name):

        input_words = open(file_name)
        word_file = open('wordsEn.txt', 'r')
        words_list = word_file.read().split()
        for word in words_list:
            self.add_word(word.rstrip())

    def add_node_to_column(self, node, column, letter):
        if column >= len(self.columns):
            self.add_column(node, letter)
        elif letter not in self.columns[column]:
            self.columns[column][letter] = [node]
        else:
            self.columns[column][letter].append(node)

    def add_column(self, node, letter):
        self.columns.append({letter: [node]})

    def letter_at(self, letter, position):
        nodes = self.columns[position][letter]
        for node in nodes:
            self.print_lexi(node)

    def __str__(self):
        self.print_lexi(self.root)
        return ''.format(end='')

    def alphabet_to_num(self, char):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        return alphabet.index(char)

    def find_matches(self, match_str):
        # Find the last fixed letter (greatly reduces complexity)
        last_index = 0
        for index, char in enumerate(match_str):
            if char != '?':
                last_index = index

        if last_index == 0:
            nodes = [self.root.next[self.alphabet_to_num(match_str[0])]]
        else:
            nodes = self.columns[last_index+1][match_str[last_index]]

        suffix_len = len(match_str)-(last_index+1)
        matched_words = []

        prefix_str = match_str[0:last_index+1][::-1]
        for node in nodes:
            if self.match_prefix(node, prefix_str):
                matched_words += self.print_lexi(node,suffix_len)
        return matched_words

    def match_prefix(self, node, prefix):
        curr_node = node
        i = 0
        prefix_len= len(prefix)
        while curr_node is not None and i < prefix_len:
            if curr_node.letter == prefix[i] or prefix[i] == '?':
                i += 1
                curr_node = curr_node.previous
            else:
                return False
        return True

    def print_lexi(self, node, word_len):
        current = node
        string_so_far = current.letter
        matched_words = []

        while node.previous is not None:
            string_so_far += node.previous.letter
            node = node.previous
        string_so_far = string_so_far[::-1]
        if current.isWord:
            if word_len == 0:
                matched_words.append(string_so_far)
        for node in current.next:
            self.__print_lexi_r(string_so_far, node, matched_words, word_len-1)

        return matched_words

    def __print_lexi_r(self, string_so_far, current, matched_words, word_len):
        if word_len >= 0:
            string_so_far += current.letter
            if current.isWord:
                if word_len == 0:
                    matched_words.append(string_so_far)
            if current.next:
                for node in current.next:
                    self.__print_lexi_r(string_so_far, node, matched_words, word_len-1)

    @staticmethod
    def look_next(letter, current_node):
        """
        Returns the next node if it exists
        """
        if not current_node.next:
            return False
        else:
            for node in current_node.next:
                if node.letter == letter:
                    return node
                else:
                    pass
            return False

    def add_word(self, word):
        index = 0
        current = self.root
        for i in range(0, len(word)):
            next_node = self.look_next(word[i], current)
            is_new_node = False
            if next_node:
                current = next_node
            else:
                new_node = Node(word[i], current)
                current.next.append(new_node)
                current = new_node
                is_new_node = True
            index += 1
            if is_new_node:
                self.add_node_to_column(current, index, word[i])
        current.isWord = True
        


if __name__ == '__main__':

    word_trie = Trie()
    word_trie.build_trie('english_words.txt')
    
    cProfile.run('word_trie.find_matches("l???a?")')
    print(word_trie.find_matches("???a?"))

