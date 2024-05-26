from graphviz import Digraph
import json


id=0
class edge:
    def __init__(self, label, source, destination):
        self.label = label
        self.source = source
        self.destination = destination

class state:
    def __init__(self, label):
        self.label = label
        self.out_edges = []

class NFA:
    def __init__(self, start, accept, inner_states):
        self.start = start
        self.accept = accept
        self.inner_states = inner_states
        
        
def preprocessOR(regex):
    alphanumeric = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    valid_chars_before = alphanumeric + [']', ')', '.', '*', '+', '?']
    valid_chars_after = alphanumeric + ['[', '(', '.']
    invalid_start_chars = ['*', '?', '+']
    invalid_preceding_chars = ['*', '?', '+', '|', '[', '(']

    if regex[0] in ['|','*','+','?']  or regex[-1] == '|':
        raise ValueError("Invalid OR in regex")

    for i in range(1, len(regex)):
        if regex[i] in invalid_start_chars and regex[i - 1] in invalid_preceding_chars:
            raise ValueError("Invalid character in regex")

        if regex[i] == '|':
            if regex[i - 1] not in valid_chars_before or regex[i + 1] not in valid_chars_after:
                raise ValueError("Invalid OR in regex")
    return regex

def preprocessRanges(regex):
    i = 0
    stack = []
    range_start = None
    while i < len(regex):
        if regex[i] == '[':
            stack.append('[')
            i += 1
        elif regex[i] == ']':
            if not stack or stack[-1] != '[':
                raise ValueError("Unbalanced brackets in regex")
            stack.pop()
            i += 1
        elif regex[i] == '-':
            if range_start is None or not regex[i+1].isalnum():
                raise ValueError(f"Invalid regex: '-' must have an alphanumeric character before it")
            if not ((range_start.isalpha() and regex[i+1].isalpha() and ((range_start.isupper() and regex[i+1].isupper()) or (range_start.islower() and regex[i+1].islower()))) or (range_start.isdigit() and regex[i+1].isdigit())):
                raise ValueError(f"Invalid regex: '-' must be between two alphanumeric characters of the same type and case")
            if range_start > regex[i+1]:
                raise ValueError(f"Invalid regex: The character before '-' must be before the character after '-'")
            range_start = None
            i += 2
        else:
            range_start = regex[i]
            i += 1
    if stack:
        raise ValueError("Unbalanced brackets in regex")
    return regex


def preprocessConcat(regex):
    result = ''
    in_brackets = False  # Flag to check if we are inside square brackets
    for i in range(len(regex) - 1):
        c = regex[i]
        v = regex[i + 1]
        if c == '[':
            in_brackets = True
        elif c == ']':
            in_brackets = False
        if not in_brackets:  # Only add '@' if we are not inside square brackets
            if c in ['*', '?', ')', '+', ']'] and v not in ['*', '?', ')', '+', '|']:
                result += c + '@'
            elif (c.isalnum() or c in ['.',']',')']) and (v.isalnum()  or v in ['(','.','[']):
                result += c + '@'
            else:
                result += c
        else:
            result += c  # If we are inside square brackets, just add the character without '@'
    result += regex[-1]
    return result

def preprocessAll(infix):
    result = preprocessConcat(preprocessRanges(preprocessOR(infix)))
    if not result:
        return()
    elif all(c in '()[]' for c in result):
        raise ValueError("Regex has only brackets")
    return result

def precedence(char):
    precedence_dict = {'|': 1, '@': 2, '?': 3, '+': 4, '*': 4, '(': 0}
    return precedence_dict.get(char, 0)

def is_operator(character):
    operators = ['|', '@', '?', '+', '*', '(']
    return character in operators

def infix_to_postfix(infix):
    postfix=''
    stack=[]
    i = 0
    while i < len(infix):
        if infix[i] == '[':               #opening square bracket
            term = '['
            i += 1
            while infix[i] != ']':
                term += infix[i]
                i += 1
                if i >= len(infix):
                    raise ValueError('Invalid regex: Unbalanced square brackets, opening without closing bracket')
            term += ']'
            postfix += term
        elif infix[i] == '(':               #opening bracket
            stack.append(infix[i])
        elif infix[i] == ')':             #closing bracket
            if not stack:
                raise ValueError('Invalid regex: Unbalanced brackets, closing without opening bracket')
            while stack[-1] != '(':
                postfix += stack.pop()
                if not stack:
                    raise ValueError('Invalid regex: Unbalanced brackets, closing without opening bracket')
            stack.pop()                 #pop the opening bracket

        elif is_operator(infix[i]):
            while stack and precedence(stack[-1]) >= precedence(infix[i]):
                postfix += stack.pop()
            stack.append(infix[i])
        else:
            postfix += infix[i]
        i += 1

    while stack:          #stack is not guaranteed empty when infix is empty
      if stack[-1] == '(':
        raise ValueError('Invalid regex: Unbalanced brackets, opening without closing bracket')
      postfix += stack.pop()

    return postfix


def concatNFA(stack):
    nfa2 = stack.pop()
    nfa1 = stack.pop()
    nfa1.accept.out_edges.append(edge('ε', nfa1.accept, nfa2.start))
    inner_states = nfa1.inner_states + nfa2.inner_states
    stack.append(NFA(nfa1.start, nfa2.accept, inner_states))

def orNFA(stack):
    global id
    nfa2 = stack.pop()
    nfa1 = stack.pop()
    start = state("S"+str(id)); id+=1
    accept = state("S"+str(id)); id+=1
    start.out_edges.append(edge('ε', start, nfa1.start))
    start.out_edges.append(edge('ε', start, nfa2.start))
    nfa1.accept.out_edges.append(edge('ε', nfa1.accept, accept))
    nfa2.accept.out_edges.append(edge('ε', nfa2.accept, accept))
    inner_states = nfa1.inner_states + nfa2.inner_states + [start, accept]
    stack.append(NFA(start, accept, inner_states))

def starNFA(stack):
    global id
    nfa = stack.pop()
    start = state("S"+str(id)); id+=1
    accept = state("S"+str(id)); id+=1
    start.out_edges.append(edge('ε', start, nfa.start))
    start.out_edges.append(edge('ε', start, accept))
    nfa.accept.out_edges.append(edge('ε', nfa.accept, start))
    nfa.accept.out_edges.append(edge('ε', nfa.accept, accept))
    inner_states = nfa.inner_states + [start, accept]
    stack.append(NFA(start, accept, inner_states))

def plusNFA(stack):
    global id
    nfa = stack.pop()
    start = state("S"+str(id)); id+=1
    accept = state("S"+str(id)); id+=1
    start.out_edges.append(edge('ε', start, nfa.start))
    nfa.accept.out_edges.append(edge('ε', nfa.accept, start))
    nfa.accept.out_edges.append(edge('ε', nfa.accept, accept))
    inner_states = nfa.inner_states + [start, accept]
    stack.append(NFA(start, accept, inner_states))

def questionNFA(stack):
    global id
    nfa = stack.pop()
    start = state("S"+str(id)); id+=1
    accept = state("S"+str(id)); id+=1
    start.out_edges.append(edge('ε', start, nfa.start))
    start.out_edges.append(edge('ε', start, accept))
    nfa.accept.out_edges.append(edge('ε', nfa.accept, accept))
    inner_states = nfa.inner_states + [start, accept]
    stack.append(NFA(start, accept, inner_states))
    
def constructNFA(c:str, stack):
    global id
    start = state("S"+str(id)); id+=1
    accept = state("S"+str(id)); id+=1
    start.out_edges.append(edge(c, start, accept))
    inner_states = [start, accept]
    stack.append(NFA(start, accept, inner_states))

def postfix_to_NFA(postfix):
    alphanumeric = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    stack = []
    in_brackets = False
    bracket_content = ''
    for c in postfix:
        if c == '[':
            in_brackets = True
            bracket_content = '['
            continue
        elif c == ']':
            in_brackets = False
            constructNFA(bracket_content+']', stack)
            continue
        if in_brackets:
            bracket_content += c
            continue
        if c == '@':
            concatNFA(stack)
        elif c == '|':
            orNFA(stack)
        elif c == '*':
            starNFA(stack)
        elif c == '+':
            plusNFA(stack)
        elif c == '?':
            questionNFA(stack)
        elif c in alphanumeric or c=='.' or c=='-':
            constructNFA(c, stack)
        else:
            print(f'Invalid character in postfix regex {c}')
    nfa = stack.pop()
    drawNFA(nfa)
    return nfa

def drawNFA(nfa):
    dot = Digraph()
    dot.attr(rankdir='LR')

    dot.node('start', shape='none', width='0')
    dot.edge('start', nfa.start.label)

    for state in nfa.inner_states:
        if state == nfa.accept:
            dot.node(state.label, peripheries='2')
        else:
            dot.node(state.label)
        for edge in state.out_edges:
            dot.edge(edge.source.label, edge.destination.label, label=edge.label)
    dot.format = 'png'
    dot.render('./compilers/nfa_output', cleanup=True)
    
def NFAtoJSON(nfa):
    visited = set()
    nodes = {}

    def dfs(state):
        if state in visited:
            return
        visited.add(state)
        nodes.update({state.label: {}})
        nodes[state.label].update({'isTerminatingState': state == nfa.accept})
        for edge in state.out_edges:
            nodes[state.label].setdefault(edge.label, [])
            nodes[state.label][edge.label].append(edge.destination.label)
            dfs(edge.destination)

    dfs(nfa.start)
    nodes.update({'startingState': nfa.start.label})
    with open('NFA.json', 'w') as f:
        json.dump(nodes, f, indent=4)
        
        
        
class Node:
    def __init__(self, value):
        self.value = value
        self.Out = {}
        self.end: Node = self
        self.acceptingState = False
        self.epsilonClosure = []
        self.acceptingepsilonClosure = False

    def copmuteEpsilonClosure(self):
        visited = set()
        def dfs(node):
            if node.value in visited:
                return
            if node.acceptingState:
                self.acceptingepsilonClosure = True
            visited.add(node.value)
            for i in node.Out.get('ε', []):
                dfs(i)
        dfs(self)
        self.epsilonClosure = frozenset(visited)
        
def jsonToNodes():
    file = json.load(open('NFA.json'))
    starting = file.pop('startingState')
    nodes = {}
    for i in file:
        x = Node(i)
        x.acceptingState = file[i].pop('isTerminatingState')
        nodes.update({i: x})

    starting = nodes[starting]
    for i in file:
        for j in file[i]:
            for k in file[i][j]:
                nodes[i].Out.setdefault(j, []).append(nodes[k])

    return starting, nodes

def precomputeEpsilonClosure(nodes):
    for i in nodes:
        nodes[i].copmuteEpsilonClosure()
        
def NFAtoDFA(nodes, starting):
    dfa = {starting.epsilonClosure: {}}
    toVisit = [starting.epsilonClosure]
    while toVisit:
        current = toVisit.pop()
        dfa.setdefault(current, {})
        for i in current:
            canBeVisited = frozenset()
            for j in nodes[i].Out:
                if j != 'ε':
                    dfa[current].setdefault(j, frozenset())
                    for k in nodes[i].Out[j]:
                        dfa[current][j] = dfa[current][j].union(k.epsilonClosure)
            if nodes[i].acceptingepsilonClosure:
                dfa[current].setdefault('isTerminatingState', True)
        for i in dfa[current]:
            if i != 'isTerminatingState':
                if dfa[current][i] not in dfa:
                    toVisit.append(dfa[current][i])
                    dfa.setdefault(dfa[current][i], {})

    return dfa

def drawDFA(dfa):
    dot = Digraph(comment='DFA')

    dot.node('', shape='none')
    dot.edge('',str(set(starting.epsilonClosure)), label='Start')

    dot.attr(rankdir='LR')

    for i in dfa:
        if 'isTerminatingState' in dfa[i]:
            dot.node(str(set(i)), shape='doublecircle')
        else:
            dot.node(str(set(i)))
        for j in dfa[i]:
            if j != 'isTerminatingState':
                dot.edge(str(set(i)), str(set(dfa[i][j])), label=j)

    dot.format = 'png'
    dot.render('dfa_output', cleanup=True)
    
    
def initialSplit(dfa):
    nodesAsKeys = dict()
    groupAsKeys = dict()
    groupAsKeys.setdefault('0', set())
    groupAsKeys.setdefault('1', set())
    for i in dfa:
        nodesAsKeys.setdefault(i, '0')
        if 'isTerminatingState' in dfa[i]:
            nodesAsKeys[i] = '1'
        groupAsKeys[nodesAsKeys[i]].add(i)
    return groupAsKeys, nodesAsKeys

def splitGroups(group, nodes, dfa, counter=2):
    newGroups = dict()
    for i in group:
        for j in dfa[i]:
            if j != 'isTerminatingState':
                newGroups.setdefault(j, dict())
                newGroups[j].setdefault(nodes[dfa[i][j]], set())
                newGroups[j][nodes[dfa[i][j]]].add(i)
    if len(newGroups) == 0:
        return group, counter
    output = set()
    for i in newGroups:
        for j in newGroups[i]:
            if len(output) == 0:
                output.add(frozenset(newGroups[i][j]))

                test1 = frozenset(group.difference(newGroups[i][j]))
                if len(test1) > 0:
                    output.add(test1)
            else:
                for k in output.copy():
                    output.discard(k)
                    test1 = k.difference(newGroups[i][j])
                    test2 = k.difference(test1)
                    if len(test1) > 0:
                        output.add(test1)
                    if len(test2) > 0:
                        output.add(test2)
    if output == {frozenset(group)}:
        return group, counter
    newGroups = dict()
    for i in output:
        for j in i:
            nodes[j] = str(counter)
        newGroups.setdefault(str(counter), set())
        newGroups[str(counter)].update(i)
        counter += 1

    return newGroups, counter

def drawMinDFA(dfa,cc):
    dot = Digraph(comment='DFA')

    dot.node('', shape='none')
    dot.edge('',str(nodes[starting.epsilonClosure]), label='Start')

    dot.attr(rankdir='LR')

    for i in dfa:
        if 'isTerminatingState' in dfa[i]:
            dot.node(i, shape='doublecircle')
        else:
            dot.node(i)
        for j in dfa[i]:
            if j != 'isTerminatingState':
                dot.edge(i, dfa[i][j], label=j)

    dot.format = 'png'
    dot.render(f'./compilers/test{cc}/min_dfa_output', cleanup=True)
    


cc = 8
# infix = ')AB('
# infix = "A|"
# infix = "[Z-A]"
# infix = "AB"
# infix = "A|B"
# infix = "(AB|[A-Z])+[A-Z]*"
# infix = "(AB|C|[A-Z]S*)+ABC"
# infix = "(((AB)((A|B)*))(AB))"
# infix = "AB(A|B)*AB"
# infix = "([A-Z])"
infix = "([A-C][A-C]|A|ABCD*C+)[B-D]"
# infix = "(a|b)*abb"
# infix = "m?[0-9]+"
# infix = "((a|b|c)+9|55?(zzz)*)"
# infix = "((a|b|c+v))"

# infix=input()

print((preprocessAll(infix)))
print(infix_to_postfix(preprocessAll(infix)))
NFAtoJSON(postfix_to_NFA(infix_to_postfix(preprocessAll(infix))))

starting, nodes = jsonToNodes()
precomputeEpsilonClosure(nodes)
dfa = NFAtoDFA(nodes, starting)
drawDFA(dfa)


groups, nodes = initialSplit(dfa)
counting = 2
while True:
    initialGroups = groups.copy()
    for i in initialGroups:
        test, counting = splitGroups(groups[i], nodes, dfa, counting)
        if test != groups[i]:
            groups.pop(i)
            groups.update(test)

    if initialGroups == groups:
        break

mindfa = dict()
done = set()
for i in nodes:
    if nodes[i] not in done:
        done.add(nodes[i])
        mindfa.setdefault(nodes[i], dict())
        for j in dfa[i]:
            if j != 'isTerminatingState':
                # mindfa[nodes[i]].setdefault('isTerminatingState', False)
                mindfa[nodes[i]].setdefault(j, nodes[dfa[i][j]])
            else:
                mindfa[nodes[i]].setdefault(j, True)
                
drawMinDFA(mindfa,cc)

mindfa.setdefault("startingState", str(nodes[starting.epsilonClosure]))
with open('MinDfa.json', 'w') as f:
        json.dump(mindfa, f, indent=4)