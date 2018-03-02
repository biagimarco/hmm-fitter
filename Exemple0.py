a = []
a.append({"b": "b1", "c": 3})
a.append({"b": "b2", "c": 1})
a.append({"b": "b3", "c": 2})

best = min(a, key=lambda x: x['c'])

print best["b"]
