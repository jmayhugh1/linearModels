def mystery_algorithm(a, b):
    x = a
    y = b
    while x != y:
        if x > y:
            x = x - y
        else:
            y = y - x
    return x  # or return y, since x == y

# Test with the values a = 2437 and b = 875
a = 2437
b = 875
result = mystery_algorithm(a, b)
print("The result is:", result)
