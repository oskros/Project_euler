from helper import *


def problem_101():
    def get_poly(y_vals):
        x_vals = list(range(1, len(y_vals) + 1))
        coefs = [round(x, 0) for x in list(lagrange(x_vals, y_vals))]
        powers = list(reversed(range(len(coefs))))
        return lambda n: sum([coef*(n**power) for coef, power in zip(coefs, powers)])

    true_func = lambda n: 1 - n + n**2 - n**3 + n**4 - n**5 + n**6 - n**7 + n**8 - n**9 + n**10
    true_seq = []
    i = 1
    out = 0
    while True:
        true_seq.append(true_func(i))

        tmp_op = get_poly(true_seq)
        if tmp_op(i+1) != true_func(i+1):
            out += tmp_op(i+1)
        else:
            break
        i += 1
    return out


def problem_103(search_range=3):
    def next_near_optimal(special_optimal_set):
        middle = special_optimal_set[len(special_optimal_set) // 2]
        return [middle] + [x + middle for x in special_optimal_set]

    def optimal_in_vicinity(near_optimal, search_range=3):
        min_sum = sum(near_optimal)
        true_optimal = near_optimal

        all_perms = itertools.product(range(-search_range, search_range + 1), repeat=len(near_optimal))
        for perm in all_perms:
            new_set = sorted([x + y for x, y in zip(near_optimal, perm)])

            if any(new_set.count(x) > 1 or x < 1 for x in new_set):
                continue
            if is_special_set(new_set):
                new_sum = sum(new_set)
                if new_sum < min_sum:
                    min_sum = new_sum
                    true_optimal = new_set
        return true_optimal

    def optimal_special_set(n, search_range=3):
        if n == 1:
            return [1]
        else:
            return optimal_in_vicinity(next_near_optimal(optimal_special_set(n - 1)), search_range)

    return ''.join(map(str, optimal_special_set(7, search_range)))


def problem_104():
    sq5 = 5**0.5

    def check_first_digits(n):
        def mypow(x, n):
            res = 1.0
            for i in range(n):
                res *= x
                # truncation to avoid overflow:
                if res > 1e20:
                    res /= 1e10
            return res

        # this is an approximation for large n:
        F = mypow((1 + sq5) / 2, n) / sq5
        s = '%f' % F
        return set(s[:9]) == set('123456789')

    a, b, n = 1, 1, 1
    while True:
        if set(str(a)[-9:]) == set('123456789'):
            if check_first_digits(n):
                return n
        a, b = b, a + b
        b = b % 1000000000
        n += 1


def problem_105():
    out = 0
    with open("files/p105_sets.txt", 'r') as fo:
        for line in fo:
            cur_set = list(map(int, line.strip().split(',')))
            if is_special_set(cur_set):
                out += sum(cur_set)
    return out


def problem_106():
    lst = range(1, 12 + 1)
    subsets = get_subsets(lst)
    out = 0
    for x_idx, x in enumerate(subsets):
        for y in subsets[x_idx + 1:]:  # start second loop after first loop to avoid double counting
            if len(x) == len(y) == 1:  # strictly increasing, so subsets of length 1 are never equal in sum
                continue
            if len(x) != len(y):  # 2nd rule is assumed to hold, so subsets of different length have different sums
                continue
            if set(x) - (set(x) - set(y)):  # not a subset pair if their intersection is non-empty
                continue
            if all(i < j for i, j in zip(x, y)):  # if subset A is pairwise lower than subset B, we don't need to check
                continue
            out += 1
    return out


def problem_107():
    # This problem is solved with Krim's algorithm
    class Krims:
        def __init__(self, matrix):
            self.matrix = matrix
            self.V = len(matrix)

        def saved_weight(self):
            parent = self.prim_mst()
            return sum(sum(self.matrix, []))//2 - sum(self.matrix[i][parent[i]] for i in range(1, self.V))

        def min_key(self, key, mst_set):
            tmp_min = float('inf')
            min_index = float('inf')
            for v in range(self.V):
                if key[v] < tmp_min and mst_set[v] is False:
                    tmp_min = key[v]
                    min_index = v
            return min_index

        def prim_mst(self):
            key = [float('inf')] * self.V
            key[0] = 0

            parent = [None] * self.V
            parent[0] = -1

            mst_set = [False] * self.V
            for _ in range(self.V):
                u = self.min_key(key, mst_set)
                mst_set[u] = True
                for v in range(self.V):
                    if 0 < self.matrix[u][v] < key[v] and mst_set[v] is False:
                        key[v] = self.matrix[u][v]
                        parent[v] = u
            return parent

    with open("files/p107_network.txt", 'r') as fo:
        input_matrix = [[int(y) if y != '-' else 0 for y in x.strip().split(',')] for x in fo.readlines()]
        saved = Krims(input_matrix).saved_weight()
    return saved


def problem_108():
    # See that 1/x + 1/y = 1/n requires that x, y > n
    # We can replace x = a + n, y = b + n, for s,r in natural numbers
    # Inserting the replacements and doing some algebra leaves us at a*b = n^2
    # Thus all solutions will be pairs of factors of n^2. However, each pair will be counted double, since we count
    # (a,b) and (b,a), so we divide by 2. However, as n^2 is a perfect square, it has an odd number of divisors, so we
    # add one before dividing by 2, in order to not exclude the "middle" divisor.
    lim = 1000
    n = 1
    primes = primes_sieve(100)
    while True:
        nod_sq = (nod_primes(n ** 2, primes) + 1) // 2
        if nod_sq > lim:
            return n
        n += 1


def problem_109():
    rng = list(map(str, range(1, 21)))
    plist = rng + ['25'] + ['D' + x for x in rng] + ['D25'] + ['T' + x for x in rng]

    def get_points(num_str):
        if num_str[0] == 'D':
            out = int(num_str[1:]) * 2
        elif num_str[0] == 'T':
            out = int(num_str[1:]) * 3
        else:
            out = int(num_str)
        return out

    checkout_1 = [(x,) for x in plist if 'D' in x]
    checkout_2 = [(y, x[0]) for x in checkout_1 for y in plist]
    checkout_3 = [(z, y, x[0]) for x in checkout_1 for y_idx, y in enumerate(plist) for z in plist[y_idx:]]

    count = 0
    for checkout in checkout_1 + checkout_2 + checkout_3:
        points = [get_points(throw) for throw in checkout]
        if sum(points) < 100:
            count += 1
    return count


def problem_110():
    limit = 4e6
    primes = primes_sieve(100)
    dio = lambda x: (nod_primes(x**2, primes) + 1) // 2
    upper_bound = []
    for prime in primes:
        upper_bound.append(prime)
        if dio(reduce(operator.mul, upper_bound)) >= limit:
            break
    while True:
        for i in range(0, len(primes)):
            tmp = primes[i:i+2] + upper_bound[:-1]
            if (limit <= dio(reduce(operator.mul, tmp)) < dio(reduce(operator.mul, upper_bound))
                    and reduce(operator.mul, tmp) < reduce(operator.mul, upper_bound)):
                upper_bound = tmp
                break
        else:
            return reduce(operator.mul, upper_bound)


def problem_111():
    num = [1]*10

    def checknum():
        if num[0] == 0:
            return 0
        n = 0
        for i in range(0, len(num)):
            n = n*10+num[i]

        rm = miller_rabin(n)
        return n if rm else 0

    def recurse(basedigit, startpos, level, fill=False):
        if level <= 0:
            return checknum()
        res = 0
        if fill:
            for pos in range(len(num)):
                num[pos] = basedigit
        for pos in range(startpos, len(num)):
            for val in range(10):
                num[pos] = val
                res += recurse(basedigit, pos+1, level-1)
                num[pos] = basedigit
        return res

    result = 0
    for d in range(10):
        for i in range(1, len(num)):
            tmp_sum = recurse(d, 0, i, True)
            if tmp_sum > 0:
                result += tmp_sum
                break

    return result


def problem_112():
    def is_bouncy(num):
        incr = None
        str_n = str(num)
        for i in range(1, len(str_n)):
            n0 = str_n[i - 1]
            n1 = str_n[i]
            if n0 == n1:
                continue
            elif n0 < n1:
                if incr is False:
                    return True
                else:
                    incr = True
            else:
                if incr is True:
                    return True
                else:
                    incr = False
        return False

    count, i = 0, 100
    ratio = float(count) / i
    while ratio < 0.99:
        i += 1
        if is_bouncy(i):
            count += 1
        ratio = float(count) / i
    return i


def problem_113():
    # No. of numbers to be printed is 100, no. of increases possible is 9 (starts at 0, so can increase by 9)
    # We thus need 100 prints, and 9 increases, leaving us with selecting 9 from 100+9
    # Subtract 1 to avoid counting 0 as a number
    increasing_numbers = binomial_coef(100 + 9, 9) - 1
    decreasing_numbers = binomial_coef(100 + 10, 10) - 1 - 100  # subtract repeating pattern zeros
    duplicates = 900
    return increasing_numbers + decreasing_numbers - duplicates


def problem_114():
    return ways_to_fill(50, 3)


def problem_115():
    tot = 0
    length = 50
    while tot < 1e6:
        length += 1
        tot = ways_to_fill(length, 50)
    return length


def problem_116():
    @Memoize
    def fill_same_blocks(length, block_size):
        count = 1

        if length < block_size:
            return count

        for position in range(length - block_size + 1):
            count += fill_same_blocks(length - position - block_size, block_size)

        return count

    return sum(fill_same_blocks(50, x) for x in [2, 3, 4])


def problem_117():
    @Memoize
    def fill_same_blocks(length, min_block_size, max_block_size):
        count = 1

        if length < min_block_size:
            return count

        for block_size in range(min_block_size, max_block_size + 1):
            for position in range(length - block_size + 1):
                count += fill_same_blocks(length - position - block_size, min_block_size, max_block_size)

        return count

    return fill_same_blocks(50, 2, 4)


def problem_118():
    @Memoize
    def check_partitions(s_idx, prev, perm):
        count = 0
        for i in range(s_idx, len(perm)):
            num = 0
            for j in range(s_idx, i + 1):
                num = num * 10 + perm[j]

            if num < prev or not is_prime_naive(num):
                continue

            if i == len(perm) - 1:
                return count + 1

            count += check_partitions(s_idx=i + 1, prev=num, perm=perm)
        return count

    rng = list(range(1, 10))
    tot_count = 0
    for perm in itertools.permutations(rng):
        tot_count += check_partitions(s_idx=0, prev=0, perm=perm)
    return tot_count


def problem_119():
    numbers_to_find = 35
    max_exponent_guess = 15
    max_digitsum_guess = 200
    out = []
    for a in range(2, max_digitsum_guess):
        val = a
        for _ in range(2, max_exponent_guess):
            val *= a
            if digit_sum(val) == a:
                out.append(val)
            if len(out) >= numbers_to_find:
                return sorted(out)[29]


def problem_120():
    r = lambda a, n: ((a-1)**n + (a+1)**n) % (a*a)
    out = 0
    for a in range(3, 1001):
        n_max = a if a % 2 == 0 else a*2
        out += max(r(a, n) for n in range(1, n_max+1))
    return out


def problem_121():
    turns = 15
    prev = [1]
    for n_red in range(1, turns + 1):
        new = [0]*(len(prev) + 1)
        for idx, elem in enumerate(prev):
            new[idx] += elem
            new[idx+1] += elem*n_red
        prev = new
    return math.factorial(turns+1) // sum(prev[:len(prev)//2])


def problem_122():
    lim = 201
    cost = [float('inf')]*lim
    path = [float('inf')]*lim

    def backtrack(power, depth):
        if power >= lim or depth > cost[power]:
            return
        cost[power] = depth
        path[depth] = power
        for j in range(depth, -1, -1):
            backtrack(power + path[j], depth + 1)

    backtrack(1, 0)
    return sum(cost[1:])


def problem_123():
    # For n odd we can reduce (p_n + 1)^n + (p_n - 1)^n mod p_n^2 to 2*p_n*n
    # For n even we can reduce to 2. Thus we know solution must be for n odd.
    lim = 10**10
    primes = primes_sieve(500000)
    n = 3
    while 2*primes[n-1]*n < lim:
        n += 2
    return n


def problem_124():
    lim = 100000
    radicals = {i: radical(i) for i in range(1, lim + 1)}
    return sorted(radicals, key=radicals.get)[10000-1]


def problem_125():
    lim = 1e8
    sq_lim = lim**0.5

    found_numbers = set()
    for i in range(1, int(sq_lim) + 1):
        num = i**2
        for j in range(i+1, int(sq_lim) + 1):
            num += j**2
            if num > lim:
                break

            if num == int(''.join(reversed(str(num)))):
                found_numbers.add(num)
    return sum(found_numbers)


def problem_127(lim=120000):
    # It follows that hcf(b, c) and hfc(c, a) must also be 1 since a + b = c
    # It also follows that radical(a*b*c) = radical(a)*radical(b)*radical(c) if hfc(a, b) == 1
    c_sum = 0
    radicals = radical_sieve(lim)

    for a in range(1, lim//2):
        for b in range(a + 1, lim - a):
            c = a + b
            if radicals[a]*radicals[b]*radicals[c] < c and hcf(a, b) == 1:
                c_sum += c
    return c_sum


def problem_129(max_n=10**6):
    def a(n):
        if hcf(n, 10) != 1:
            return 0
        k = x = 1
        while x != 0:
            x = (x * 10 + 1) % n
            k += 1
        return k

    i = max_n+1
    while a(i) <= max_n:
        i += 2
    return i


def problem_130():
    def a(n):
        k = x = 1
        while x != 0:
            x = (x * 10 + 1) % n
            k += 1
        return k

    lst = []
    i = 1
    while len(lst) < 25:
        i += 2
        if hcf(i, 10) == 1 and not is_prime_naive(i) and (i-1) % a(i) == 0:
            lst.append(i)
    return sum(lst)


def problem_131():
    def is_perfect_cube(x):
        x = abs(x)
        return int(round(x ** (1. / 3))) ** 3 == x

    n_primes = 1000000
    primes = primes_sieve(n_primes)

    out = 0
    cur_cube = 0
    for p in primes:
        tmp_cube = cur_cube + 1
        for _ in range(15):
            n = tmp_cube ** 3
            if is_perfect_cube(n**3+n**2*p):
                out += 1
                cur_cube = tmp_cube
                break
            tmp_cube += 1

    return out


def problem_132():
    # We find that R(k) = (10^k - 1)/9, thus we can calculate R(k) % p = 10^k % 9p.
    # This can efficiently be done with the modpow function, as pow(10, k, p)
    primes = iter(primes_sieve(200000))
    pfs = []
    while len(pfs) < 40:
        cur_prime = next(primes)
        modpow = pow(10, 10**9, 9*cur_prime)
        if modpow == 1:
            pfs.append(cur_prime)
    return sum(pfs)


def problem_134(lim=10**6):
    primes = primes_sieve(lim+3)[2:]

    out = 0
    for idx, p in enumerate(primes[:-1]):
        p1 = p
        p2 = primes[idx+1]

        i = 1
        while int(str(i) + str(p1)) % p2 != 0:
            i += 1
        out += int(str(i) + str(p1))
    return out


def problem_135():
    # n = x^2 - y^2 - z^2 = (z + 2d)^2 - (z + d)^2 - z^2 = 3d^2 + 2dz - z^2 = (3d - z)(d + z) = u*v
    # d = (u + v)/4 and z = (3v-u)/4
    # we know d and z must be positive integers
    lim = 1000000
    sols = [0]*(lim+1)
    for u in range(1, lim+1):
        for v in range(1, lim//u+1):
            if (u + v) % 4 == 0 and 3*v > u and (3*v-u) % 4 == 0:
                sols[u*v] += 1
                # d = (u+v)//4
                # z = (3*v-u)//4
                # y = z+d
                # x = z+2*d
                # res = "%s**2 - %s**2 - %s**2 == %s" % (x, y, z, u*v)
                # print(res)
                # print(eval(res))

    return len([x for x in sols if x == 10])


def problem_136():
    # n = x^2 - y^2 - z^2 = (z + 2d)^2 - (z + d)^2 - z^2 = 3d^2 + 2dz - z^2 = (3d - z)(d + z) = u*v
    # d = (u + v)/4 and z = (3v-u)/4
    # we know d and z must be positive integers
    lim = 50000000
    sols = [0]*(lim+1)
    for u in range(1, lim+1):
        for v in range(1, lim//u+1):
            if (u + v) % 4 == 0 and 3*v > u and (3*v-u) % 4 == 0:
                sols[u*v] += 1

    return len([x for x in sols if x == 1])


def problem_144():
    def next_intersect(x0, y0, x1, y1):
        euclidean_dist = (-4*x1 * -4*x1 + y1 * y1) ** 0.5
        norm_x1 = -4*x1 / euclidean_dist
        norm_y1 = -y1 / euclidean_dist
        direction_x = x1 - x0
        direction_y = y1 - y0
        reflect_x = direction_x - 2 * (direction_x * norm_x1 + direction_y * norm_y1) * norm_x1
        reflect_y = direction_y - 2 * (direction_x * norm_x1 + direction_y * norm_y1) * norm_y1

        a = reflect_y / reflect_x

        x2 = (4 * x1 - a * a * x1 + 2 * a * y1) / (-4 - a * a)
        y2 = a * (x2 - x1) + y1

        return x2, y2

    from_x, from_y = 0.0, 10.1
    to_x, to_y = 1.4, -9.6
    count = 0
    while to_x < -0.01 or to_x > 0.01 or to_y < 9.9:
        tmp = next_intersect(from_x, from_y, to_x, to_y)
        from_x, from_y = to_x, to_y
        to_x, to_y = tmp[0], tmp[1]
        count += 1
    return count


def problem_145(lim=10**9):
    continue_check()

    def is_reversible(n):
        num = n
        if n % 10 == 0:
            return False
        rev = 0
        while num > 0:
            rev = rev * 10 + num % 10
            num //= 10
        rev += n
        while rev > 0:
            if (rev % 10) % 2 == 0:
                return False
            rev //= 10
        return True

    count = 0
    for i in range(1, lim):
        if is_reversible(i):
            count += 1
    return count


def problem_146():
    lim = int(1.5 * 10**8)
    out = 0
    for n in range(10, lim, 10):
        n2 = n * n
        if n2 % 3 != 1:
            continue
        if n2 % 7 not in [2, 3]:
            continue
        if n2 % 9 == 0 or n2 % 13 == 0 or n2 % 27 == 0:
            continue
        if miller_rabin(n2 + 1) and miller_rabin(n2 + 3) and miller_rabin(n2 + 7) and miller_rabin(
                        n2 + 9) and miller_rabin(n2 + 13) and miller_rabin(n2 + 27) and not miller_rabin(
                        n2 + 19) and not miller_rabin(n2 + 21):
            out += n
    return out


def problem_149():
    @Memoize
    def s(k):
        if 1 <= k <= 55:
            return (100003 - 200003 * k + 300007 * k ** 3) % 1000000 - 500000
        elif 56 <= k <= 4000000:
            return (s(k - 24) + s(k - 55) + 1000000) % 1000000 - 500000

    table = [[s(i * 2000 + k) for i in range(2000)] for k in range(1, 2001)]

    def kadane_1d(lst):
        out = tot = lst[0]
        for i in range(1, len(lst)):
            tot = max(tot + lst[i], lst[i])
            out = max(out, tot)
        return out

    def get_backward_diagonals(grid):
        b = [None] * (len(grid) - 1)
        grid = [b[i:] + r + b[:i] for i, r in enumerate([[c for c in r] for r in grid])]
        return [[c for c in r if c is not None] for r in zip(*grid)]

    def get_forward_diagonals(grid):
        b = [None] * (len(grid) - 1)
        grid = [b[:i] + r + b[i:] for i, r in enumerate([[c for c in r] for r in grid])]
        return [[c for c in r if c is not None] for r in zip(*grid)]

    max_cols = max(kadane_1d(col) for col in table)
    max_rows = max([kadane_1d([row[x] for row in table]) for x in range(len(table[0]))])
    max_diag1 = max([kadane_1d(x) for x in get_backward_diagonals(table)])
    max_diag2 = max([kadane_1d(x) for x in get_forward_diagonals(table)])

    return max(max_cols, max_rows, max_diag1, max_diag2)


if __name__ == '__main__':
    print(problem_146())
