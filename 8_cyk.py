def cyk_algorithm(input_string, grammar):
    # Convert grammar to a dictionary where keys are non-terminals and values are sets of right-hand side symbols.
    grammar_dict = {}
    for lhs, rhs in grammar:
        if lhs not in grammar_dict:
            grammar_dict[lhs] = set()
        grammar_dict[lhs].add(tuple(rhs))




    # Length of the input string
    n = len(input_string)




    # Create a table to store non-terminals for substrings
    P = [[set() for _ in range(n)] for _ in range(n)]




    # Step 1: Fill the diagonal (base case)
    for i in range(n):
        for lhs, rhs in grammar_dict.items():
            for rule in rhs:
                if rule == (input_string[i],):
                    P[i][i].add(lhs)




    # Step 2: Fill the rest of the table (dynamic programming)
    for length in range(2, n + 1): # length of the substring
        for i in range(n - length + 1): # start of the substring
            j = i + length - 1 # end of the substring
            for k in range(i, j): # split point
                for lhs, rhs in grammar_dict.items():
                    for rule in rhs:
                        if len(rule) == 2 and rule[0] in P[i][k] and rule[1] in P[k+1][j]:
                            P[i][j].add(lhs)




    # Step 3: Check if the start symbol can generate the entire string
    start_symbol = grammar[0][0] # Assuming the first production gives the start symbol
    return start_symbol in P[0][n-1]




# Updated grammar for the sentence "the dog eating the fish"
grammar = [
    ('S', ['NP', 'VP']),
    ('NP', ['Det', 'N']),
    ('VP', ['V', 'NP']),
    ('VP', ['V', 'N']), # New rule for verb + noun (eating + fish)
    ('Det', ['the']),
    ('N', ['dog']),
    ('N', ['fish']),
    ('V', ['eating']),
]


input_string = "the dog eating the fish"
result = cyk_algorithm(input_string.split(), grammar)
print(f"Can the input string be generated? {'Yes' if result else 'No'}")
























# Practical 8: CYK Algorithm


def cyk_algorithm(input_string, grammar):
    grammar_dict = {}
    for lhs, rhs in grammar:
        if lhs not in grammar_dict:
            grammar_dict[lhs] = set()
        grammar_dict[lhs].add(tuple(rhs))

    n = len(input_string)
    P = [[set() for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for lhs, rhs in grammar_dict.items():
            for rule in rhs:
                if rule == (input_string[i],):
                    P[i][i].add(lhs)

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for lhs, rhs in grammar_dict.items():
                    for rule in rhs:
                        if len(rule) == 2 and rule[0] in P[i][k] and rule[1] in P[k+1][j]:
                            P[i][j].add(lhs)

    start_symbol = grammar[0][0]
    return start_symbol in P[0][n - 1]

grammar = [
    ('S', ['NP', 'VP']),
    ('NP', ['Det', 'N']),
    ('VP', ['V', 'NP']),
    ('VP', ['V', 'N']),
    ('Det', ['the']),
    ('N', ['dog']),
    ('N', ['fish']),
    ('V', ['eating'])
]

input_string = "the dog eating the fish"
result = cyk_algorithm(input_string.split(), grammar)
print("\n--- CYK Algorithm ---")
print(f"Can the input string be generated? {'Yes' if result else 'No'}")
print("Input string tokens:", input_string.split())







