def argmax(ls):
    if not ls:
        return None, 0.0
    return max(ls, key = lambda x: x[1])
