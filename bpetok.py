def token(lst):
    dic = {}
    for i in range(0,len(lst)-1):
        dic[(lst[i],lst[i+1])] = dic.get((lst[i], lst[i+1]),0)+1
    return dic

def merge(lst, tup):
    nlst = []
    stri = tup[0]+tup[1]
    i = 0
    merge = False
    while i < len(lst)-1:
        if lst[i] == tup[0] and lst[i+1] == tup[1]:
            if i == len(lst)-2: merge = True
            nlst.append(stri)
            i += 1
        else:
            nlst.append(lst[i])
        i += 1
    if not merge:
        nlst.append(lst[len(lst)-1])
    return nlst

def bpe(vocab, n=1000):
    lst = list(vocab)
    vlst = []
    for i in range(n):
        dic = token(lst)
        tup = max(dic,key=dic.get)
        nlst = merge(lst,tup)
        vlst.append(tup)
        lst = nlst
    return vlst

def build_vocab(rules):
    dic = {}
    i=0
    while i < 256:
        dic[chr(i)] = i
        i += 1
    for rule in rules:
        stri = rule[0]+rule[1]
        dic[stri] = i
        i += 1
    return dic

def encode(text, rules, vocab):
    lst = list(text)
    for rule in rules:
        lst = merge(lst,rule)
    for i in range(0,len(lst)):
        lst[i] = vocab[lst[i]]
    return lst

def decode(lst):
    word = ""
    for cha in lst:
        word += cha
    return word

corpus = "aaabdaaabac"
rules = bpe(corpus, n=5)
vocab = build_vocab(rules)
encoded = encode("aaab", rules, vocab)
decoded = decode([r[0]+r[1] for r in rules[:3]])
print(rules)
print(vocab)
print(encoded)