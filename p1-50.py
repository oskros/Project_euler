from helper import *


def problem_1():
    return sum(x for x in range(1000) if x % 3 == 0 or x % 5 == 0)


def problem_2():
    out = 0
    a, b = 1, 1
    while b < 4e6:
        a, b = b, a + b
        if b % 2 == 0:
            out += b
    return out


def problem_3():
    return max(prime_factors(600851475143))


def problem_4():
    return max([x*y for x in range(1, 1000) for y in range(1, 1000) if x*y == int(str(x*y)[::-1])])


def problem_5():
    return next(x for x in range(2520, 10**10, 20) if all(x % i == 0 for i in range(19, 0, -1)))


def problem_6():
    return sum(range(1, 101)) ** 2 - sum(x ** 2 for x in range(1, 101))


def problem_7():
    return primes_sieve(200000)[10000]


def problem_8():
    num_string = """73167176531330624919225119674426574742355349194934969835203127745063262395783180169848018694788518
    438586156078911294949545950173795833195285320880551112540698747158523863050715693290963295227443043557668966489504
    452445231617318564030987111217223831136222989342338030813533627661428280644448664523874930358907296290491560440772
    390713810515859307960866701724271218839987979087922749219016997208880937766572733300105336788122023542180975125454
    059475224352584907711670556013604839586446706324415722155397536978179778461740649551492908625693219784686224828397
    224137565705605749026140797296865241453510047482166370484403199890008895243450658541227588666881164271714799244429
    282308634656748139191231628245861786645835912456652947654568284891288314260769004224219022671055626321111109370544
    217506941658960408071984038509624554443629812309878799272442849091888458015616609791913387549920052406368991256071
    76060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450"""\
        .replace(' ', '').replace('\n', '')
    products = []
    for i in range(1000):
        substring = num_string[i:i + 13]
        prod = 1
        for elem in substring:
            prod *= int(elem)

        products.append(prod)
    return max(products)


def problem_9():
    return next(a*b*(1000-a-b) for a in range(1, 333) for b in range(a, 1000) if a+b+(a*a+b*b)**0.5 == 1000)


def problem_10():
    return sum(primes_sieve(2000000))


def problem_11():
    mat = """08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48""".split('\n')
    mat = np.array([np.array(list(map(int, x.split(' ')))) for x in mat])

    max_row = np.max(mat[:, :-3] * mat[:, 1:-2] * mat[:, 2:-1] * mat[:, 3:])
    max_col = np.max(mat[:-3, :] * mat[1:-2, :] * mat[2:-1, :] * mat[3:, :])
    max_rdg = np.max(mat[:-3, :-3] * mat[1:-2, 1:-2] * mat[2:-1, 2:-1] * mat[3:, 3:])
    max_ldg = np.max(mat[:-3, 3:] * mat[1:-2, 2:-1] * mat[2:-1, 1:-2] * mat[3:, :-3])

    return max(max_row, max_col, max_rdg, max_ldg)


def problem_12():
    trinum = 0
    i = 1
    primes = primes_sieve(100)
    while nod_primes(trinum, primes) < 500:
        trinum += i
        i += 1
    return trinum


def problem_13():
    with open('files/p013_sums.txt') as fo:
        nums = [int(x.strip()) for x in fo.readlines()]
    return int(str(sum(nums))[:10])


def problem_14():
    seqs = {1: 1}
    for i in range(1, 1000000):
        n = i
        steps = 0

        # tmplst = [(i, 0)]
        while n not in seqs:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3*n+1
            steps += 1
            # tmplst.append((n, steps))
        # for t in tmplst:
        #     seqs[t[0]] = steps - t[1] + seqs[n]
        seqs[i] = seqs[n] + steps

    return max(seqs, key=seqs.get)


def problem_15():
    return int(binomial_coef(40, 20))


def problem_16():
    return sum(int(x) for x in str(2**1000))


def problem_17():
    specials = {0: 0, 1: 3, 2: 3, 3: 5, 4: 4, 5: 4, 6: 3, 7: 5, 8: 5, 9: 4,
                10: 3, 11: 6, 12: 6, 13: 8, 14: 8, 15: 7, 16: 7, 17: 9, 18: 8, 19: 8}
    decades = {2: 6, 3: 6, 4: 5, 5: 5, 6: 5, 7: 7, 8: 6, 9: 6}
    centuries = {0: 0, 1: 10, 2: 10, 3: 12, 4: 11, 5: 11, 6: 10, 7: 12, 8: 12, 9: 11}

    count = 11  # onethousand = 11
    for c in range(10):
        for d in range(10):
            for y in range(10):
                tmp = centuries[c]
                if c > 0 and d+y > 0:
                    tmp += 3
                if d < 2:
                    tmp += specials[d*10+y]
                else:
                    tmp += decades[d] + specials[y]
                count += tmp
    return count


def problem_18():
    t = [[75],
         [95, 64],
         [17, 47, 82],
         [18, 35, 87, 10],
         [20, 4, 82, 47, 65],
         [19, 1, 23, 75, 3, 34],
         [88, 2, 77, 73, 7, 63, 67],
         [99, 65, 4, 28, 6, 16, 70, 92],
         [41, 41, 26, 56, 83, 40, 80, 70, 33],
         [41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
         [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
         [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
         [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
         [63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
         [4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23]]

    for i in range(len(t)-1, 0, -1):
        for j in range(len(t[i])-1):
            t[i-1][j] += max(t[i][j], t[i][j+1])
    return t[0][0]


def problem_19():
    sundays = 0
    for year in range(1901, 2001):
        for month in range(12):
            if dt.date(year, month+1, 1).weekday() == 6:
                sundays += 1
    return sundays


def problem_20():
    return sum(int(x) for x in str(math.factorial(100)))


def problem_21():
    total = 0
    for i in range(1, 10000):
        tmp = sum_of_divisors(i)
        if sum_of_divisors(tmp) == i and i != tmp:
            total += i
    return total


def problem_22():
    with open('files/p022_names.txt', 'r') as fo:
        names = sorted(fo.readline().replace('"', '').split(','))

    return sum(i*sum(ord(l)-64 for l in name) for i, name in enumerate(names, 1))


def problem_23():
    abundant = [i for i in range(1, 28123) if sum_of_divisors(i) > i]

    soa = [False] * 28124
    for i, a1 in enumerate(abundant):
        for a2 in abundant[i:]:
            if a1+a2 > 28123:
                break
            soa[a1+a2] = True
    return sum(i for i, b in enumerate(soa) if b is False)


def problem_24():
    tmp = itertools.permutations(list(range(10)))
    for _ in range(1000000-1):
        next(tmp)
    return int(''.join(map(str, next(tmp))))


def problem_25():
    a, b = 1, 1
    i = 3
    while len(str(a+b)) < 1000:
        a, b = b, a + b
        i += 1
    return i


def problem_26():
    max_len = 0
    for n in range(2, 1001):
        exponent = 1
        mod = 1
        if n % 2 > 0 and n % 5 > 0:
            while mod > 0:
                mod = (10**exponent - 1) % n
                exponent += 1
            max_len = max(max_len, exponent)
    return max_len


def problem_27():
    max_conseq = 0
    max_prod = 0
    sieve = BoolPrimesSieve(15000)

    for a in range(-999, 1000):
        for b in range(-1000, 1001):
            n = 0
            while sieve.bool_is_prime(n**2 + a*n + b):
                n += 1

            if n > max_conseq:
                max_conseq = n
                max_prod = a*b
    return max_prod


def problem_28():
    def gen_points(end):
        _moves = itertools.cycle([lambda x, y: (x, y - 1), lambda x, y: (x - 1, y),
                                  lambda x, y: (x, y + 1), lambda x, y: (x + 1, y)])
        n = 1
        pos = 0, 0
        times_to_move = 1

        yield n, pos
        while True:
            for _ in range(2):
                move = next(_moves)
                for _ in range(times_to_move):
                    if n >= end:
                        return
                    pos = move(*pos)
                    n += 1
                    yield n, pos

            times_to_move += 1

    return sum([x[0] for x in list(gen_points(1001*1001)) if abs(x[1][0]) == abs(x[1][1])])


def problem_29():
    return len({a**b for a in range(2, 101) for b in range(2, 101)})


def problem_30():
    return sum(i for i in range(2, 500000) if i == sum(int(x)**5 for x in str(i)))


def problem_31():
    total = 200
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    num_ways = [1] + [0]*total
    for coin in coins:
        for intermediate_sums in range(total-coin+1):
            num_ways[coin+intermediate_sums] += num_ways[intermediate_sums]
    return num_ways[total]


def problem_32():
    return sum({a*b for a in range(1, 100) for b in range(1000//a, 9999//a) if ''.join(sorted(str(a) + str(b) + str(a*b))) == '123456789'})


def problem_33():
    num, denom = 1, 1
    for a in range(10, 100):
        for b in range(a+1, 100):
            if int(str(b)[-1]) != 0 and a/b == int(str(a)[0]) / int(str(b)[-1]) and int(str(a)[1]) == int(str(b)[0]):
                num *= a
                denom *= b

    return denom//hcf(num, denom)


def problem_34():
    return sum(i for i in range(4, 50000) if i == sum(math.factorial(int(x)) for x in str(i)))


def problem_35():
    plist = [x for x in primes_sieve(10**6) if x < 10 or all(n not in str(x) for n in ['2', '4', '5', '6', '8', '0'])]
    out = 0
    for p in plist:
        str_p = str(p)
        perms = [str_p[i:] + str_p[:i] for i in range(1, len(str_p)+1)]
        if all(rabin_miller(int(x)) for x in perms):
            out += 1
    return out


def problem_36():
    return sum(n for n in range(10**6) if n == int(str(n)[::-1]) and bin(n)[2:] == bin(n)[2:][::-1])


def problem_37():
    primes = [x for x in primes_sieve(10**6) if x > 10]
    return sum(p for p in primes if all(rabin_miller(p%10**i) and rabin_miller(p//10**i) for i in range(1, len(str(p)))))


def problem_38():
    out = 0
    for i in range(2, 10**5):
        tmp = str(i)
        for j in range(2, 10):
            if len(tmp) == 9:
                break
            tmp += str(i*j)
        if ''.join(sorted(tmp)) == '123456789':
            out = max(out, int(tmp))
    return out


def problem_39():
    out = {p: sum(1 for b in range(1, p//2) if (p*p-2*p*b) % (2*p-2*b) == 0) for p in range(3, 1001)}
    return max(out, key=out.get)


def problem_40():
    irr = ''.join(str(x) for x in range(10**6+1))
    return reduce(lambda x, y: x*y, [int(irr[10**x]) for x in range(7)])


def problem_41():
    """
    Any number is congruent with the sum of its digits modulo 9. Thus, if the sum of digits is 3, 6, or 0 (mod 9) the
    number is divisible by 3. Which means the number is either 3 or composite.
    """
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    while sum(digits) % 9 in [0, 3, 6]:
        digits.pop(-1)
    perms = [int(''.join(map(str, x))) for x in itertools.permutations(digits)]

    while not rabin_miller(perms[-1]):
        perms.pop(-1)
    return perms[-1]


def problem_42():
    tri_nums = 0
    with open('files/p042_words.txt', 'r') as fo:
        words = fo.readline().replace('"', '').split(',')
    for w in words:
        lsum = sum(ord(l)-64 for l in w)

        if (1 + 8*lsum)**0.5 % 2 == 1:
            tri_nums += 1
    return tri_nums


def problem_43():
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    divs = [17, 13, 11, 7, 5, 3, 2]

    mod_dict = dict()
    for poss in itertools.permutations(digits, 3):
        for perm in divs:
            if int(''.join(map(str, poss))) % perm == 0:
                mod_dict.setdefault(perm, []).append(list(poss))

    possibilities = mod_dict[17]
    for i in range(1, 7):
        reduced_possibilities = []
        for poss in possibilities:
            for prev_perm in mod_dict[divs[i]]:
                if poss[:2] == prev_perm[-2:] and prev_perm[0] not in poss:
                    reduced_possibilities.append([prev_perm[0]] + poss)
        possibilities = reduced_possibilities

    return sum(int(''.join(map(str, list(set(digits) - set(p)) + p))) for p in possibilities)


def problem_44():
    i = 0
    while True:
        i += 1
        for j in range(i-1, 0, -1):
            pi, pj = polygonal(5, i), polygonal(5, j)
            if (1+24*(pi-pj))**0.5 % 6 == 5 and (1+24*(pi+pj))**0.5 % 6 == 5:
                return pi-pj


def problem_45():
    x = 286
    while True:
        c = (x ** 2 + x) / 2
        y = (1 + (1 + 24*c)**0.5) / 6
        z = (1 + (1 + 8*c)**0.5) / 4

        if y == int(y) and z == int(z):
            return x
        x += 1


def problem_46():
    sieve = primes_sieve(10**6)
    odds = list(range(3, 10**6, 2))
    comp = list(set(odds) - set(sieve))

    for c in comp:
        t = 0
        for p in sieve:
            n = (c-p)/2
            if n < 1:
                break
            if n**0.5 % 1 == 0:
                t = n
                break
        if t == 0:
            return c


def problem_47():
    conseq = 0
    i = 643
    while conseq < 4:
        i += 1
        if len(set(prime_factors(i))) == 4:
            conseq += 1
        else:
            conseq = 0
    return i-3


def problem_48():
    return sum(pow(x, x, 10**10) for x in range(1, 1001)) % 10**10


def problem_49():
    primes = [p for p in primes_sieve(10000) if p > 1487 and '0' not in str(p)]
    for i, p1 in enumerate(primes):
        for j, p2 in enumerate(primes[i+1:]):
            if sorted(str(p1)) == sorted(str(p2)):
                p3 = 2*p2-p1
                if rabin_miller(p3) and sorted(str(p3)) == sorted(str(p2)) and p3 < 10000:
                    return int(str(p1)+str(p2)+str(p3))


def problem_50():
    max_n = int(1e6)

    def max_chain(x, primes):
        rsum, i = 0, 0
        while rsum <= x:
            rsum += primes[i]
            i += 1
        return i

    prime_lst = primes_sieve(max_n)
    lst = []
    innerloop_max = max_chain(max_n, prime_lst)
    for i in range(4):
        for x in range(i, innerloop_max):
            tempsum = sum(prime_lst[i:x])
            if is_prime_naive(tempsum) and tempsum < max_n: lst.append((x - i, tempsum))
    return max(lst)


if __name__ == '__main__':
    print(problem_27())
