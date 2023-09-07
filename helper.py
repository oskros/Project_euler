import itertools
import copy
import tqdm
import time
import timeit
import math
import random
import bisect
import sys
import numpy as np
import sympy as sy
import functools
import operator
import atexit
import datetime as dt
from functools import reduce
from scipy.interpolate import lagrange
from collections import defaultdict
from decimal import Decimal, getcontext


class TimeIt:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start = timeit.default_timer()
        result = self.func(*args, **kwargs)
        elapsed = timeit.default_timer() - start

        if elapsed > 60:
            print('%r  %s min, %.2f sec' % (self.func.__name__, int(elapsed//60), elapsed % 60))
        elif elapsed > 2:
            print('%r  %.3f sec' % (self.func.__name__, elapsed))
        elif elapsed > 0.002:
            print('%r  %.3f ms' % (self.func.__name__, elapsed*1e3))
        elif elapsed > 2e-6:
            print('%r  %.3f µs' % (self.func.__name__, elapsed*1e6))
        else:
            print('%r  %.3f ns' % (self.func.__name__, elapsed*1e9))
        return result


@atexit.register
def end_time():
    final_time = timeit.default_timer() - start_time
    if final_time > 60:
        print('Time elapsed: %s min, %.2f sec' % (int(final_time//60), final_time % 60))
    elif final_time > 2:
        print('Time elapsed: %.3f sec' % final_time)
    elif final_time > 0.002:
        print('Time elapsed: %.3f ms' % (final_time*1e3))
    elif final_time > 2e-6:
        print('Time elapsed: %.3f µs' % (final_time*1e6))
    else:
        print('Time elapsed: %.3f ns' % (final_time*1e9))


class Memoize:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]


def jacobi(a, p):
    '''Computes the Jacobi symbol (a|p), where p is a positive odd number.
    :see: https://en.wikipedia.org/wiki/Jacobi_symbol
    '''
    # https://pypi.python.org/pypi/primefac
    if (p % 2 == 0) or (p < 0): return None  # p must be a positive odd number
    if (a == 0) or (a == 1): return a
    a, t = a % p, 1
    while a != 0:
        while not a & 1:
            a //= 2
            if p & 7 in (3, 5): t *= -1
        a, p = p, a
        if (a & 3 == 3) and (p & 3) == 3:
            t *= -1
        a %= p
    return t if p == 1 else 0


@Memoize
def is_prime_naive(n):
    if (n % 2 == 0 and n > 2) or n < 2:
        return False
    return all(n % i for i in range(3, int(n**0.5) + 1, 2))


def binomial_coef(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def concat_list(lst_):
    out = ''
    for n in lst_:
        out += str(n)
    return int(out)


def sum_of_divisors(x):
    s = 1
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            s += i
            if x != i**2:
                s += x // i
    return s


def primes_sieve(limit, non_primes=False):
    limit += 1
    not_prime = set()
    primes = []

    for i in range(2, limit):
        if i in not_prime:
            continue

        for f in range(i*i, limit, i):
            not_prime.add(f)

        primes.append(i)

    if non_primes:
        return not_prime, primes
    return primes


class LookupSieve:
    def __init__(self, lim):
        self.lim = lim
        self.sieve = self.build_sieve()

    def is_prime(self, x):
        if not x & 1:
            return x == 2
        return self.sieve[x >> 1]

    def build_sieve(self):
        half = self.lim >> 1
        sieve = [True]*half
        sieve[0] = False
        i = 1
        while 2*i*i < half:
            if sieve[i]:
                current = 3*i + 1
                while current < half:
                    sieve[current] = False
                    current += 2*i + 1
            i += 1
        return sieve


def decimal_to_base(decimal, base):
    hex_str = ''
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:base]
    if base > len(digits):
        raise ValueError(f"Base: {base} not supported, max base is {len(digits)}")
    if decimal == 0:
        return '0'
    elif decimal < 0:
        neg = True
        decimal *= -1
    else:
        neg = False
    while decimal != 0:
        hex_str = digits[decimal % base] + hex_str
        decimal = decimal // base
    return '-' + hex_str if neg else hex_str


def polygonal(degree, n):
    if degree == 3:
        return n*(n+1)//2
    elif degree == 4:
        return n*n
    elif degree == 5:
        return n*(3*n-1)//2
    elif degree == 6:
        return n*(2*n-1)
    elif degree == 7:
        return n*(5*n-3)//2
    elif degree == 8:
        return n*(3*n-2)
    else:
        raise NotImplementedError


def cubes_with_len(n):
    out = []
    x, cube = 0, 0
    while len(str(cube)) <= n:
        cube = x**3
        if len(str(cube)) == n:
            out.append(cube)
        x += 1
    return out


def contains_same_digits(n1, n2):
    sort_n1 = ''.join(sorted(str(n1)))
    sort_n2 = ''.join(sorted(str(n2)))

    return sort_n1 == sort_n2


def cont_frac(s, degree):
    m = 0
    d = 1
    a = s ** 0.5 // 1

    for _ in range(degree):
        m = d * a - m
        d = (s - m ** 2) / d
        a = ((s ** 0.5 + m) / d) // 1
    return a, d, m


def digit_sum(n):
    r = 0
    while n > 0:
        r += n % 10
        n //= 10
    return r


def all_digit_sums(n):
    r = 0
    while n < 0:
        r += n % 10
        n //= 10
    return r


def all_subdigit_groups(str_n):
    if not str_n:
        return [[]]
    if len(str_n) == 1:
        return [[str_n]]
    if len(str_n) == 2:
        return [[str_n], list(str_n)]

    out = [[str_n]]
    for i in range(len(str_n), 0, -1):
        tmp = str_n[:i]
        for sd in all_subdigit_groups(str_n[i:]):
            out.append([tmp] + sd)

    return out


def all_subdigit_group_sums(n):
    return [sum(map(int, grp)) for grp in all_subdigit_groups(str(n))]


def totient(num):
    result = num
    p = 2
    while p * p <= num:
        if num % p == 0:
            while num % p == 0:
                num //= p
            result -= result // p
        p += 1

    if num > 1:
        result -= result // num
    return result


def totient_sieve(limit):
    limit += 1
    phi = list(range(1, limit))

    for p in range(2, limit):
        if phi[p-1] == p:
            phi[p-1] = p - 1
            for i in range(2*p, limit, p):
                phi[i-1] = (phi[i-1]//p) * (p-1)
    return phi


@Memoize
def hcf(no1, no2):
    while no1 != no2:
        if no1 > no2:
            no1 -= no2
        elif no2 > no1:
            no2 -= no1
    return no1


def pentag_range_below(n):
    start = 1
    out = []
    next_2 = [1, 2]
    while next_2[0] <= n:
        if next_2[-1] <= n:
            out.extend(next_2)
        else:
            out.append(next_2[0])
        start += 1
        next_2 = [polygonal(5, y*start) for y in [1, -1]]
    return out


class AStarSearch:
    def __init__(self, matrix):
        self.matrix = matrix
        self.result = self.a_star()

    def a_star(self):
        dim = len(self.matrix)

        start = (0, 0)
        goal = (dim - 1, dim - 1)

        closed_set = set()
        open_set = {start}

        came_from = defaultdict(lambda: None)

        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0

        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = self._matrix_cost_estimate(start, goal)

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            if current == goal:
                return self._path_sum(came_from, current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self._get_neighbors(current, dim):
                if neighbor in closed_set:
                    continue

                if neighbor not in open_set:
                    open_set.add(neighbor)

                tentative_gscore = g_score[current] + self.matrix[neighbor[0]][neighbor[1]]
                if tentative_gscore >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_gscore
                f_score[neighbor] = g_score[neighbor] + self._matrix_cost_estimate(neighbor, goal)

        return "failure"

    def _matrix_cost_estimate(self, start, end):
        min_steps = sum(end) - sum(start) + 1
        ordered = sorted(sum(self.matrix, []))
        return sum(ordered[:min_steps])

    def _path_sum(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)

        return sum([self.matrix[x[0]][x[1]] for x in total_path])

    @staticmethod
    def _get_neighbors(current, dim):
        out = []
        if current[0] > 0:
            out.append((current[0] - 1, current[1]))
        if current[1] > 0:
            out.append((current[0], current[1] - 1))
        if current[0] < dim - 1:
            out.append((current[0] + 1, current[1]))
        if current[1] < dim - 1:
            out.append((current[0], current[1] + 1))
        return out


class SudokuSolver:
    def __init__(self, grid, unassigned=0):
        self.grid = copy.deepcopy(grid)
        self.unassigned = unassigned
        self.operations = 0
        self.solver()

    def valid_grid(self, i, j, e):
        # Check row, column and subgrid
        if e not in [self.grid[i][x] for x in range(9)]:
            if e not in [self.grid[x][j] for x in range(9)]:
                if e not in [self.grid[x][y] for y in range(j//3*3, j//3*3 + 3) for x in range(i//3*3, i//3*3 + 3)]:
                    return True

        return False

    def solver(self):
        i, j = next(((x, y) for x in range(0, 9) for y in range(0, 9) if self.grid[x][y] == self.unassigned), (-1, -1))
        if i == -1:
            return True
        for e in range(1, 10):
            if self.valid_grid(i, j, e):
                self.operations += 1
                self.grid[i][j] = e
                if self.solver():
                    return True
                # Undo the current cell for backtracking
                self.grid[i][j] = 0
        return False


def conseq_integers(input_list):
    idx_count = 0
    while input_list[idx_count] == idx_count + 1:
        idx_count += 1
    return idx_count


def integer_to_roman2(integer):
    value_map = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    char_order = sorted(value_map, key=value_map.get, reverse=True)
    out = ''

    tmp_int = integer
    for idx, roman in enumerate(char_order):
        val = value_map[roman]

        out += roman*(tmp_int // val)
        tmp_int %= val

        legal_sub = char_order[(idx + 2) // 2 * 2] if roman != 'I' else None
        if legal_sub is not None:
            sub_mod = tmp_int % (val - value_map[legal_sub])
            if tmp_int > sub_mod:
                out += legal_sub
                out += roman

                tmp_int = sub_mod
    return out


def integer_to_roman(integer):
    value_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                 (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    out = ''

    for val, roman in value_map:
        out += roman * (integer // val)
        integer %= val

    return out


def roman_to_integer(roman):
    value_map = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    out = 0
    prev = 0

    for cur in reversed(roman):
        value = value_map[cur]
        out += value if value >= prev else -value
        prev = value

    return out


def len_of_factorial_chain(num):
    known = [num]
    count = 1
    while True:
        if count > 60:
            return 61
        num = sum(math.factorial(int(x)) for x in str(num))
        if num in known:
            break
        known.append(num)
        count += 1
    return count


class MonopolyOdds:
    def __init__(self, max_dice_no=4):
        self.max_dice_no = max_dice_no
        self.CC_tiles = [2, 17, 33]
        self.CH_tiles = [7, 22, 36]
        self.GO = 0
        self.G2J = 30
        self.JAIL = 10

        self.CC_cards = list(range(16))
        self.CH_cards = list(range(16))

    def run(self, iterations):
        result = [0]*40

        random.shuffle(self.CC_cards)
        random.shuffle(self.CH_cards)

        double_count = 0
        tile = 0

        for _ in range(iterations):
            die1 = random.randint(1, self.max_dice_no)
            die2 = random.randint(1, self.max_dice_no)
            if die1 == die2:
                double_count += 1
            else:
                double_count = 0

            if double_count == 3:
                double_count = 0
                tile = self.JAIL
            else:
                tile += die1 + die2
                if tile >= 40:
                    tile -= 40
                if tile == self.G2J:
                    tile = self.JAIL
                if tile in self.CH_tiles:
                    tile = self.draw_CH(tile=tile)
                if tile in self.CC_tiles:
                    tile = self.draw_CC(tile=tile)
            result[tile] += 1
        return result

    def draw_CC(self, tile):
        card = self.CC_cards.pop(0)
        self.CC_cards.append(card)

        if card == 0:
            return self.GO
        elif card == 1:
            return self.JAIL
        else:
            return tile

    def draw_CH(self, tile):
        card = self.CH_cards.pop(0)
        self.CH_cards.append(card)

        if card == 0:
            return self.GO
        elif card == 1:
            return self.JAIL
        elif card == 2:
            return 11
        elif card == 3:
            return 24
        elif card == 4:
            return 39
        elif card == 5:
            return 5
        elif card in [6, 7]:
            lst = [5, 15, 25, 35]
            return next((x for x in lst if x > tile), lst[0])
        elif card == 8:
            lst = [12, 28]
            return next((x for x in lst if x > tile), lst[0])
        elif card == 9:
            return tile - 3
        else:
            return tile


def factors(n):
    step = 2 if n % 2 else 1
    return set(functools.reduce(list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1, step) if n % i == 0)))


def prime_factors(n):
    i = 2
    pf = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            pf.append(i)
    if n > 1:
        pf.append(n)
    return pf


def nod_naive(n):
    count = 0
    int_root = int(n**0.5)
    for i in range(1, int_root + 1):
        if n % i == 0:
            count += 2

    if int_root * int_root == n:
        count -= 1
    return count


def nod_primes(n, primes):
    nod = 1
    remain = n
    for prime in primes:
        if prime * prime > n:
            return nod
        exp = 1
        while remain % prime == 0:
            exp += 1
            remain //= prime
        nod *= exp
        if remain == 1:
            return nod
    return nod


@Memoize
def radical(n):
    i = 2
    pf = set()
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            pf.add(i)
    if n > 1:
        pf.add(n)

    return reduce(operator.mul, pf, 1)


def radical_sympy(n):
    fac = sy.factorint(n)
    return reduce(operator.mul, fac.keys())


def radical_sieve(limit):
    result = [1] * (limit + 1)
    result[0] = 0
    for i in range(2, len(result)):
        if result[i] == 1:
            for j in range(i, len(result), i):
                result[j] *= i
    return result


@Memoize
def multiplicative_partitions(num):
    @Memoize
    def factors_w_len(cur_num, length):
        if length == 1:
            return [(cur_num,)]
        tmp_res = set()
        for f in range(2, int(cur_num ** 0.5) + 1):
            if cur_num % f == 0:
                for j in factors_w_len(cur_num=cur_num / f, length=length - 1):
                    tmp_res.add(tuple(sorted([f] + [int(x) for x in j])))
        return tmp_res

    res, length = [], 1
    while True:
        c_factors = factors_w_len(cur_num=num, length=length)
        if not c_factors:
            return res
        res.extend(c_factors)
        length += 1


def int_sqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def word_number_mapping(word, number):
    word_to_num, num_to_word = {}, {}
    for idx in range(len(str(number))):
        (char, num) = word[idx], str(number)[idx]
        if char in word_to_num and word_to_num[char] != num:
            return None
        if num in num_to_word and num_to_word[num] != char:
            return None
        word_to_num[char] = num
        num_to_word[num] = char
    return word_to_num


def tetration(a, b, modulo):
    last = 0
    res = 1
    while b > 0:
        res = pow(a, res, modulo)
        if last == res:
            break
        b -= 1
    return res


def get_subsets(lst):
    out = []
    for i in range(1, len(lst)):
        tmp = itertools.combinations(lst, i)
        out.extend(list(tmp))
    return out


def lowest_largest_sums(lst):
    sorted_list = sorted(lst)
    for i in range(2, len(sorted_list)):
        low = sorted_list[:i]
        high = sorted_list[-len(low) + 1:]
        if sum(high) > sum(low):
            return False
    return True


def is_special_set(lst):
    def no_duplicate_sums(subsets):
        tmp = [sum(x) for x in subsets]
        return all(tmp.count(x) == 1 for x in tmp)

    if no_duplicate_sums(get_subsets(lst)) and lowest_largest_sums(lst):
        return True
    return False


@Memoize
def ways_to_fill(length, min_block_size=3):
    count = 1  # Start count at 1, to include the empty case

    for position in range(length - min_block_size + 1):
        for block_size in range(min_block_size, length - position + 1):
            count += ways_to_fill(length - position - block_size - 1, min_block_size)

    return count


def set_partitions(lst):
    if len(lst) == 1:
        yield lst
        return

    tmp = lst[0]
    for elem in set_partitions(lst[1:]):
        for idx, subset in enumerate(elem):
            yield elem[:idx] + [tmp + subset] + elem[idx+1:]
        yield [tmp] + elem


def ordered_set_partitions(lst):
    if not lst:
        return []

    out = []
    for idx in range(len(lst)-1):
        out.append([lst[:idx+1]] + [lst[idx+1:]])
        for part in ordered_set_partitions(lst[idx + 1:]):
            out.append([lst[:idx + 1]] + part)

    return out


def prime_partitions(n):
    if n < 2:
        return set()
    primes = tuple(primes_sieve(n))

    @Memoize
    def recurse_partitions(no, primes):
        if no == 2:
            return {(no,)}
        res = set()
        for prime in primes:
            sub = no-prime
            if sub in primes:
                res.add(tuple(sorted([sub, prime])))
            if sub < 2:
                continue
            sub_primes = tuple(x for x in primes if x <= sub)
            for part in recurse_partitions(sub, sub_primes):
                res.add(tuple(sorted((prime,) + part)))
        return res

    out = recurse_partitions(n, primes)
    if n == primes[-1]:
        out.add((n,))
    return out


def palindromic_numbers_below(n):
    def create_palindrome(inp, b, is_odd):
        n = inp
        palindrome = inp

        if is_odd:
            n //= b

        while n > 0:
            palindrome = palindrome * b + n % b
            n //= b
        return palindrome

    out = []
    for j in range(2):
        i = 1
        num = create_palindrome(i, 10, j % 2)
        while num < n:
            out.append(num)
            i += 1
            num = create_palindrome(i, 10, j % 2)
    return out


def integers_on_circle(n):
    # Number of integer coordinates on a circle going through (0,0), (0,N), (N,0) and (N,N) if radius=False.
    # If radius=True, then number of integer coordinates on circle with radius N
    # Proof of this is seen in 3blue1browns video about lattice points (pi in prime numbers)
    @Memoize
    def chi(x):
        tmp = x % 4
        if tmp == 3:
            return -1
        if tmp == 1:
            return 1
        return 0

    pf_dict = sy.factorint(n)
    out = 4
    for key in pf_dict:
        out *= sum(chi(key)**power for power in range(pf_dict[key] + 1))
    return out


def find_lattice_patterns():
    @Memoize
    def get_attempts(p1, p2, p3, p4, p5):
        print(p1, p2, p3, p4, p5)
        curnum = 5**p1 * 13**p2 * 17**p3 * 29**p4 * 37**p5
        ioc = integers_on_circle(curnum)

        if ioc == 420:
            return True
        elif ioc > 420:
            return False
        else:
            if p5 < p4:
                get_attempts(p1, p2, p3, p4, p5+1)
            if p4 < p3:
                get_attempts(p1, p2, p3, p4+1, p5)
            if p3 < p2:
                get_attempts(p1, p2, p3+1, p4, p5)
            if p2 < p1:
                get_attempts(p1, p2+1, p3, p4, p5)
            get_attempts(p1+1, p2, p3, p4, p5)
        return False

    get_attempts(0, 0, 0, 0, 0)
    return [key[0] for key, value in get_attempts.cache.items() if value is True]


def nongaussian_prime_sieve(limit):
    """
    Returns all nongaussian primes (primes which are not complex primes) below a given limit, and a list of numbers
    which are not multiples of the calculated nongaussian primes
    """
    limit += 1
    not_prime = set()
    primes = []
    remove = set()

    for i in range(2, limit):
        if i in not_prime:
            continue

        for f in range(i*i, limit, i):
            not_prime.add(f)

        if i % 4 == 1:
            primes.append(i)
            for f in range(2*i, limit, i):
                remove.add(f)
        else:
            not_prime.add(i)

    return sorted(list(not_prime - remove)), primes


@Memoize
def miller_rabin(n):
    if n < 2:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    if n in witnesses:
        return True
    if n % 6 not in [1, 5]:
        return False
    r, s = 1, n-1
    while s % 2 == 0:
        s //= 2
        r += 1
    for witness in witnesses:
        remainder = pow(witness, s, n)
        if remainder == 1:
            continue
        for pow_of_2 in range(1, r):
            if remainder == n - 1:
                break
            remainder = pow(remainder, 2, n)
        else:
            return False
    return True


start_time = timeit.default_timer()


if __name__ == '__main__':
    pass

