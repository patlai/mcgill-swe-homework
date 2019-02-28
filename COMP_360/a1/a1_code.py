import numpy as np
import string
import matplotlib.pyplot as plt
from graphviz import Digraph
from random import randint
import queue 

class Node:

    def __init__(self, data, weight = 0):
        self.left = None
        self.right = None
        self.data = data
        self.weight = weight
        
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(" ", self.data),
        if self.right:
            self.right.PrintTree()

def CalculateEntropy(p):
    return np.sum([-x * np.log2(x) for x in p])

def MorseCharacterToBinary(morseString):
    """ Converts a single morse character consisting of dots and dashes into its corresponding binary encoding
    """
    binaryString = ''
    for i in range(0,len(morseString)):
        if morseString[i] is '.': binaryString += '1'
        if morseString[i] is '-': binaryString += '111'
        if (i != len(morseString) - 1):
            binaryString += '0'
    binaryString += '000'
    return binaryString

def CreateMorseTree(morseAlphabet, frequencyDict):
    """ Builds a binary tree given a morse alphabet in binary encoding and a dictionary of each letter's frequency
    """
    tree = NewBlankNode()
    
    for letter in morseAlphabet:
        currentNode = tree
        for digit in letter[1][:-1]:
            if digit is '0':
                if currentNode.left is None:
                    currentNode.left = NewBlankNode()
                currentNode = currentNode.left
            if digit is '1':
                if currentNode.right is None:
                    currentNode.right = NewBlankNode()
                currentNode = currentNode.right
        last = letter[1][-1]
        # get the frequency of the letter from the frequency dictionary
        nodeData = Node([letter[0], frequencyDict[letter[0]]])
        if last is '0':
            currentNode.left = nodeData
        if last is '1':
            currentNode.right = nodeData
            
    return tree

def FindSplitIndexSF(sortedFreq):
    """ Finds the index at which to split the halves for Shannon-Fano encoding
    """
    n = len(sortedFreq)
    totalSum = np.sum([x[1] for x in sortedFreq])
    currentSum = 0
    previousSum = 0
    # loop through the sorted frequencies and find the halfway sum point
    # which occurs at either the point where the accumulated sum
    # is greated than the total sum / 2 or right before that
    # then return the index of the "halfway" point
    for i in range(0, n):
        previousSum = currentSum
        currentSum += sortedFreq[i][1]
        if (currentSum >= totalSum / 2 and i > 0):
            #currentSum = currentSum if np.abs(totalSum/2 - currentSum) < np.abs(totalSum/2 - previousSum) else previousSum 
            return i+1 if np.abs(totalSum/2 - currentSum) < np.abs(totalSum/2 - previousSum) else i
            #return i+1 if i == 0 else i
            
def NewBlankNode():
    """ create a new blank node by generating a random number in a really large range
    (since graph viz requires unique IDs for nodes -> each blank needs a different ID)
    """
    return Node(randint(0, 1000000000))
    
def ConstructTreeSF(sortedFreq):    
    if (len(sortedFreq) == 0):
        return NewBlankNode()
    
    # recurse until either 1 or 2 letter-frequency pairs are left
    if len(sortedFreq) <= 2:
        # if only 1 is left, don't create a new level -> set current node as a leaf
        if len(sortedFreq) == 1:
            return Node(sortedFreq[0])
        
        root = NewBlankNode()
        root.left = Node(sortedFreq[0])
        root.right = Node(sortedFreq[1])
        return root
        
    index = FindSplitIndexSF(sortedFreq)
    firstHalf = sortedFreq[0: index]
    secondHalf = sortedFreq[index: len(sortedFreq)]
    
    firstHalfFrequency = np.sum([x[1] for x in firstHalf])
    secondHalfFrequency = np.sum([x[1] for x in secondHalf])
    # print(" ", [x[0] for x in firstHalf])
    # print(" ", [x[0] for x in secondHalf])
    # print(" ", firstHalfFrequency)
    # print(" ", secondHalfFrequency)
    # print(' ')
    
    firstHalfTree = NewBlankNode()
    secondHalfTree = NewBlankNode()
    
    if (len(firstHalf) > 0):
        firstHalfTree = ConstructTreeSF(firstHalf) 
    if (len(secondHalf) > 0):
        secondHalfTree = ConstructTreeSF(secondHalf) 
    
    root = NewBlankNode()
    root.left = firstHalfTree
    root.right = secondHalfTree
    
    return root
    
def TreeToGraphViz(root, title):
    """ Converts a recursively defined binary tree structure into a visual graph
    """
    g = Digraph(title)
    TreeToGraphVizRecursive(root, None, g)
    return g
    
def TreeToGraphVizRecursive(root, parentData, graph, goLeft = False):
    #print(" ", root.data)
    nodeLabel = '<<font color="red" face="boldfontname">' + (' ' if isinstance(root.data, int) else str(root.data[0])) + '</font>>';
    #nodeLabel = ' ' if isinstance(root.data, int) else "%s: %.3f" % (root.data[0], root.data[1])
    nodeData = str(root.data if isinstance(root.data, int) else str(root.data[0]))
    graph.node(nodeData, label = nodeLabel,)
    
    if(parentData != None):
        graph.edge(parentData, nodeData, label = '0' if goLeft else '1')
    
    if (root.left != None):
        TreeToGraphVizRecursive(root.left, nodeData, graph, True)
        
    if (root.right != None):
        TreeToGraphVizRecursive(root.right, nodeData, graph, False)
    
def AverageBitsPerSymbol(root):
    """ Calculates the average bits per symbol of an alphabet given a binary tree
    whose leaf nodes consist of tuples of (letter, frequency) by using level order traversal
    """
    q = queue.Queue()
    q.put(root)
    averageBits = 0.0
    freq = 0.0
    level = 0
    
    # do a level order traversal on the tree
    # the average bits per symbol is the cumulative sum of level * frequency
    # for each letter in the tree
    while not q.empty():
        currentLevelCount = q.qsize()
        for i in range(0, currentLevelCount):
            node = q.get()
            if not isinstance(node.data, int):
                averageBits += node.data[1] * level
                freq += node.data[1]
                #print(" ", level)
                #print(" ", averageBits)

            if node.left: q.put(node.left)    
            if node.right: q.put(node.right)
        level += 1
        
    return averageBits

def CreateHuffmanEncoding(sortedFreq):
    trees = [Node(x, x[1]) for x in sortedFreq]
    # make a tree out of the two trees of the smallest weight while there exists more than 1 tree
    while len(trees) >= 2:
        trees = sorted(trees, key = lambda x: x.weight)
        #t1 has a smaller weight than t2
        t1 = trees.pop(0)
        t2 = trees.pop(0)
        newTree = NewBlankNode()
        newTree.weight = t1.weight + t2.weight
        newTree.left = t1
        newTree.right = t2
        trees.append(newTree)
    
    # if there is only 1 tree left in the set: return it
    return trees[0]
    

f = np.genfromtxt('original_frequencies.txt', delimiter = ' ')[:,1]
letters = list(string.ascii_lowercase)
letters.append(' ')
blankFrequency = 1/20
f = np.append(f, 1/20)
f = f / np.sum(f)

frequencies = list(zip(letters, f))
sortedFrequencies = sorted(frequencies, key = lambda x: x[1], reverse = True)
frequencyDict = dict(sortedFrequencies)

# f is the frequency of the alphabet including the blank
print("Alphabet entropy: %.4f" % CalculateEntropy(f))

morse = np.genfromtxt('morse.txt', delimiter = '\t', dtype='|U32')[:, 0:2]
morse =  sorted(morse, key = lambda x: ord(x[0]))
for m in morse:
    m[1] = MorseCharacterToBinary(m[1])

morse.append([' ', '0000'])
morseTree = CreateMorseTree(morse, frequencyDict)
morseTreeGraph = TreeToGraphViz(morseTree, 'Morse')
morseTreeGraph.view()

print("Morse average bits: %.4f" % AverageBitsPerSymbol(morseTree))

tree = ConstructTreeSF(sortedFrequencies)
g = TreeToGraphViz(tree, 'SF')
g.view()
print("Shannon-Fano average bits: %.4f" % AverageBitsPerSymbol(tree))

huffmanTree = CreateHuffmanEncoding(sortedFrequencies)
huffmanTreeGraph = TreeToGraphViz(huffmanTree, 'Huffman')
huffmanTreeGraph.view()
print("Huffman average bits: %.4f" % AverageBitsPerSymbol(huffmanTree))
