from local_derivatives import central_difference
from scalar_value_graph import Value

'''
I'm gonna build a deterministic graph of Value objects, and then traverse it in reverse order to compute gradients.

graph: $$L=\tanh(a b+c)+a^2+\log(1+e^b).$$
'''

a = Value(2.0, label='a')
b = Value(0.0, label='b')
c = Value(-1.0, label='c')

D = a * b
E = D + c
F = E.tanh()
G = a**2
H = (Value(1.0) + b.exp()).log()

L = F + G + H
L.grad = 1.0  # seed the gradient of the output node

print(f"L = {L.data}")
print(f"a = {a.data}, b = {b.data}, c = {c.data}")
print(f"da = {a.grad}, db = {b.grad}, dc = {c.grad}")

print("\nComputing gradients using reverse traversal...")
L.backward()

print(f"L = {L.data}")
print(f"a = {a.data}, b = {b.data}, c = {c.data}")
print(f"da = {a.grad}, db = {b.grad}, dc = {c.grad}")