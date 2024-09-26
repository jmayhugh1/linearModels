def gamewinner(colors):
    wendy_moves = 0
    bob_moves = 0

    i = 0
    while i < len(colors):
        j = i
        color = 0
        while j < len(colors) and colors[j] == colors[i]:
            j += 1
            color += 1
        if color > 2:
            if colors[i] == 'W':
                wendy_moves += color - 2
            else:
                bob_moves += color - 2
        i = j

    if bob_moves >= wendy_moves:
        return "bob"
    else:
        return "wendy"