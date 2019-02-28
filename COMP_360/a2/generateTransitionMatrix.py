import numpy as np
import networkx as nx

G = nx.DiGraph()
for c in 'abcdefghij':
    G.add_node(c)
    
edges = np.genfromtxt('edges_ja.csv', delimiter = ',', dtype = 'str')
for edge in edges:
    G.add_edge(edge[0], edge[1])

N = len(G.nodes)

# J is the reset matrix for which every entry is 0.15 / N
J = np.full ((N, N), (0.15 / N))
# H is the hack matrix
# -> H[i,j] is 0.85/N if page j has no hyperlinks leaving it and 0 otherwise
H = np.zeros((N, N))
# Q is the hyperlink matrix
# -> Q[i,j] is 0 if there are no hyperlinks from pj to pi and 0.85 / d+j otherwise
#.   where d+j is the number of hyperlinks leaving pj
Q = np.zeros((N, N))
             
P = np.zeros((N, N))

# links going out from each page
dplus = np.zeros(N)
for edge in G.edges:
    dplus[ord(edge[0]) - 97] += 1

for i in range (0, N):
    for j in range (0, N):
        H[i, j] = 0.85/N if dplus[j] == 0 else 0
        if G.has_edge(chr(j + 97), chr(i + 97)):
            Q[i, j] = 0.85/dplus[j]

P = J + H + Q
             
#P' = P-I
P_prime = P - np.eye(N)

# add a row of ones on top of P;
P_prime = np.concatenate((np.ones((1, 10)), P_prime), axis = 0)
np.savetxt('p_prime_ja.csv', P_prime, delimiter = ',')