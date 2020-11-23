from helper import *


def problem_205():
    pres = [0]*37
    cres = [0]*37

    for p1 in range(1, 7):
        for p2 in range(1, 7):
            for p3 in range(1, 7):
                for p4 in range(1, 7):
                    for p5 in range(1, 7):
                        for p6 in range(1, 7):
                            pres[p1+p2+p3+p4+p5+p6] += 1
    for c1 in range(1, 5):
        for c2 in range(1, 5):
            for c3 in range(1, 5):
                for c4 in range(1, 5):
                    for c5 in range(1, 5):
                        for c6 in range(1, 5):
                            for c7 in range(1, 5):
                                for c8 in range(1, 5):
                                    for c9 in range(1, 5):
                                        cres[c1+c2+c3+c4+c5+c6+c7+c8+c9] += 1
    out = 0
    for p in range(0, 37):
        for c in range(p+1, 37):
            out += cres[c]*pres[p]
    return round(out/(4**9*6**6), 7)


def problem_206():
    for a1 in reversed(range(10)):
        for a2 in reversed(range(10)):
            for a3 in reversed(range(10)):
                for a4 in range(10):
                    for a5 in range(10):
                        for a6 in range(10):
                            for a7 in range(10):
                                for a8 in range(10):
                                    for a9 in range(10):
                                        num = int('1%s2%s3%s4%s5%s6%s7%s8%s9%s0' % (a1, a2, a3, a4, a5, a6, a7, a8, a9))
                                        sq_num = num ** 0.5
                                        if int(sq_num) == sq_num:
                                            if int_sqrt(num)**2 == num:
                                                return int(sq_num)


def problem_233(lim=10**11):
    """
    There are 5 patterns giving 420 integer solutions, these have been found with trial and error using the 5 first
    non-gaussian primes (5, 13, 17, 29, 37) and the recursive search function "find_lattice_patterns" function in
    helper.py which identifies number of integer points on a circle passing through (0,0), (0,N), (N,0) and (N,N).
    The identified patterns are:
    1) p_1^3 * p_2^2 * p_3^1   * 2^b_0 * q_1^b_1 * q_2^b_2 .... < lim
    2) p_1^7 * p_2^3           * 2^b_0 * q_1^b_1 * q_2^b_2 .... < lim
    3) p_1^10 * p_2^2          * 2^b_0 * q_1^b_1 * q_2^b_2 .... < lim
    4) p_1^17 * p_2^1          * 2^b_0 * q_1^b_1 * q_2^b_2 .... < lim
    5) p_1^52                  * 2^b_0 * q_1^b_1 * q_2^b_2 .... < lim
    Here c is in {0, 1, 2, ...}, p_i is a non-gaussian prime, q_i is a gaussian prime and b_i is an even number

    Note that in case the limit is 10^11, solution 4) and 5) will be disregarded, since the smallest non-gaussian primes
    are 5 and 13, and 5^52 > 5^17*13 > 10^11

    To solve the problem, we define the gaussian prime sieve which returns 2 lists, lets call them 'P' and 'NPM'.
    'P' contains all primes below the limit which are NOT gaussian primes, and 'NPM' contains all numbers below limit
    which are not a multiple of elements in 'P'.
    We can then iterate through sets of primes in 'P' for each case described above to get the minimum solution for that
    specific set of primes, and add any extra solutions by multiplying with each element in 'NPM', while checking that
    the multiple does not exceed the limit
    """
    def add_extra_sols(prime, not_prime_multiples):
        out = prime
        for q in not_prime_multiples:
            if prime * q > lim:
                break
            out += prime * q
        return out

    # The largest prime we could need looking at the 5 solutions above, is p < lim/(p_1^3 * p_2^2)
    # where p_1 and p_2 are the two smallest non-gaussian primes (5 and 13)
    prime_lim = lim // (5**3*13**2) + 1
    npm, primes = gaussian_prime_sieve(prime_lim)
    tot_sum = 0

    # Solution 1)
    for p1 in primes:
        p1_3 = p1**3
        if p1_3 > lim:
            break
        for p2 in primes:
            if p1 == p2:
                continue
            p1p2 = p1_3 * p2**2
            if p1p2 > lim:
                break
            for p3 in primes:
                if p1 == p3 or p2 == p3:
                    continue
                p1p2p3 = p1p2 * p3
                if p1p2p3 > lim:
                    break
                tot_sum += add_extra_sols(p1p2p3, npm)

    # Solution 2)
    for p1 in primes:
        p1_7 = p1 ** 7
        if p1_7 > lim:
            break
        for p2 in primes:
            if p1 == p2:
                continue
            p1p2 = p1_7 * p2**3
            if p1p2 > lim:
                break
            tot_sum += add_extra_sols(p1p2, npm)

    # Solution 3)
    for p1 in primes:
        p1_10 = p1 ** 10
        if p1_10 > lim:
            break
        for p2 in primes:
            if p1 == p2:
                continue
            p1p2 = p1_10 * p2**2
            if p1p2 > lim:
                break
            tot_sum += add_extra_sols(p1p2, npm)

    # Solution 4)
    for p1 in primes:
        p1_17 = p1 ** 17
        if p1_17 > lim:
            break
        for p2 in primes:
            if p1 == p2:
                continue
            p1p2 = p1_17 * p2
            if p1p2 > lim:
                break
            tot_sum += add_extra_sols(p1p2, npm)

    # Solution 5)
    for p1 in primes:
        p1_52 = p1 ** 52
        if p1_52 > lim:
            break
        tot_sum += add_extra_sols(p1_52, npm)

    return tot_sum


def problem_234(lim=999966663333):
    sqlim = lim**0.5
    primes = primes_sieve(int(sqlim))

    def lps_ups(n):
        sq_n = n**0.5
        if int(sq_n) == sq_n and is_prime_naive(sq_n):
            return sq_n, sq_n
        else:
            lps = bisect.bisect_left(primes, sq_n) - 1
            return primes[lps], primes[lps+1]

    tot = 0
    for i in range(4, lim + 1):
        l, u = lps_ups(i)
        if bool(i % l) != bool(i % u):
            tot += i
    return tot


def problem_243():
    primes = primes_sieve(100)
    lim = 15499 / 94744
    current = 1
    while True:
        popped = primes.pop(0)
        current *= popped
        ratio = totient(current) / (current - 1)
        if ratio < lim:
            current //= popped
            break

    for i in range(1, popped):
        new_ratio = totient(current * i) / (current * i - 1)
        if new_ratio < lim:
            return current*i


def problem_301():
    """
    def is_nimsum(n):
        return n ^ 2*n ^ 3*n

    print([(x, bin(x), is_nimsum(x)) for x in range(10)])
    >>> [(0, '0b0', 0),
    >>>  (1, '0b1', 0),
    >>>  (2, '0b10', 0),
    >>>  (3, '0b11', 12),
    >>>  (4, '0b100', 0),
    >>>  (5, '0b101', 0),
    >>>  (6, '0b110', 24),
    >>>  (7, '0b111', 28),
    >>>  (8, '0b1000', 0),
    >>>  (9, '0b1001', 0)]

    By analysing when this is equal to zero, it can be deduced if there are no consecutive ones in the binary string,
    we get a zero nim-sum (this is a consequence of the logic in the bitwise XOR operator). Thus we can recursively
    define all bitstrings up to 2**30 and count them
    """
    @Memoize
    def recurse_bits(length, prev):
        count = 0
        if length == 0:
            return 1
        if prev == 0:
            count += recurse_bits(length - 1, 1)
        count += recurse_bits(length - 1, 0)
        return count

    return recurse_bits(30, 0)


def problem_323():
    """
    # MC Approach
    mc_lim = 10**6
    def steps():
        x = 0

        goal = 2**32-1
        count = 0
        while x != goal:
            y = random.randint(0, goal)
            x = x | y
            count += 1
        return count

    running_sum = 0
    for _ in range(mc_lim):
        running_sum += steps()
    return running_sum / mc_lim
    """
    out = 0
    tmp = 1
    x = 0
    while tmp > 10 ** -11:
        tmp = 1 - (1 - 0.5 ** x) ** 32
        out += tmp
        x += 1
    return round(out, 10)


def problem_345():
    # test_m = np.array(
    #     [[7,    53, 183, 439, 863],
    #      [497, 383, 563,  79, 973],
    #      [287,  63, 343, 169, 583],
    #      [627, 343, 773, 959, 943],
    #      [767, 473, 103, 699, 303]])

    matrix_str = """ 7  53 183 439 863 497 383 563  79 973 287  63 343 169 583
                    627 343 773 959 943 767 473 103 699 303 957 703 583 639 913
                    447 283 463  29  23 487 463 993 119 883 327 493 423 159 743
                    217 623   3 399 853 407 103 983  89 463 290 516 212 462 350
                    960 376 682 962 300 780 486 502 912 800 250 346 172 812 350
                    870 456 192 162 593 473 915  45 989 873 823 965 425 329 803
                    973 965 905 919 133 673 665 235 509 613 673 815 165 992 326
                    322 148 972 962 286 255 941 541 265 323 925 281 601  95 973
                    445 721  11 525 473  65 511 164 138 672  18 428 154 448 848
                    414 456 310 312 798 104 566 520 302 248 694 976 430 392 198
                    184 829 373 181 631 101 969 613 840 740 778 458 284 760 390
                    821 461 843 513  17 901 711 993 293 157 274  94 192 156 574
                     34 124   4 878 450 476 712 914 838 669 875 299 823 329 699
                    815 559 813 459 522 788 168 586 966 232 308 833 251 631 107
                    813 883 451 509 615  77 281 613 459 205 380 274 302  35 805"""
    matrix = np.array([[int(y) for y in x.split(' ') if len(y) > 0] for x in matrix_str.split('\n')])
    dim = len(matrix)

    def best_remaining_solution(col, used_rows):
        available = matrix[[x for x in range(dim) if x not in used_rows], col:]
        return max(sum(available.max(axis=1)), sum(available.max(axis=0)))

    global best_sol
    best_sol = max(matrix.trace(), matrix[:, ::-1].trace())  # Max diagonal sum as initial benchmark

    def solve(col, cur_sum, used_rows):
        global best_sol
        if col >= dim:
            best_sol = max(best_sol, cur_sum)
            return
        rows = set(range(dim)) - used_rows
        for r in rows:
            cur_val = matrix[r, col]
            if cur_sum + cur_val + best_remaining_solution(col, used_rows) < best_sol:
                continue
            solve(col+1, cur_sum+cur_val, used_rows.union({r}))

    solve(0, 0, set())
    return best_sol


def problem_346(lim=10**12):
    units = {1}
    for i in range(2, int(lim**0.5)+1):
        cur = 1+i+i*i
        pows = i*i
        while cur < lim:
            units.add(cur)
            pows *= i
            cur += pows
    return sum(units)


def problem_347(lim=10**3):
    def M(p, q, N):
        t = 0
        a = 1
        while p**a * q < N:
            b = 1
            while p**a * q**b <= N:
                t = max(t, p**a * q**b)
                b += 1
            a += 1
        return t

    sieve = primes_sieve(lim//2)
    out = 0
    for idx, p in enumerate(sieve):
        for q in sieve[idx+1:]:
            if p*q > lim:
                break
            out += M(p, q, lim)
    return out


def problem_357(max_num=10**8):
    sieve = BoolPrimesSieve(max_num+1)
    out = 1
    for n in range(2, max_num+1, 4):
        if not sieve.bool_is_prime(1 + n):
            continue
        for d in range(2, int(n**0.5) + 1):
            if n % d != 0:
                continue
            if not sieve.bool_is_prime(int(d + n/d)):
                break
        else:
            out += n
    return out


def problem_381():
    def modfact(n, p):
        if p <= n:
            return 0
        res = p - 1
        for i in range(n + 1, p):
            res = (res * pow(i, p - 2, p)) % p
        return res

    def s(p):
        m5 = modfact(p-5, p)
        m4 = (m5 * (p-4)) % p
        m3 = (m4 * (p-3)) % p
        m2 = (m3 * (p-2)) % p
        m1 = p - 1
        return (m1 + m2 + m3 + m4 + m5) % p

    primes = primes_sieve(10**8)[2:]
    return sum(s(p) for p in primes)


def problem_387(no_digits=14):
    """
    Idea: Start by generating harshad numbers from the left, by adding another number. Before creating a new number,
    check if the old one is strong harshad. Add every new number created if its a prime and the old one is strong
    harshad. If the new number is not prime, check if it is harshad (and by definition it will also be right truncatable
    harshad) - if it is, continue recursion with this number. Return when number is above limit.
    """
    def harshad(n):
        try:
            return n % digit_sum(n) == 0
        except ZeroDivisionError:
            return False

    def strong_harshad(n):
        try:
            return rabin_miller(n//digit_sum(n))
        except ZeroDivisionError:
            return False

    def make_prime_strong_rtrunc_harshads(length, prev_num):
        out_lst = []
        if length > 0:
            is_strong_harshad = strong_harshad(prev_num)
            for i in range(10):
                tmp = prev_num * 10 + i
                if is_strong_harshad and rabin_miller(tmp):
                    out_lst.append(tmp)
                elif harshad(tmp):
                    out_lst += make_prime_strong_rtrunc_harshads(length-1, tmp)
        return out_lst

    return sum(make_prime_strong_rtrunc_harshads(length=no_digits, prev_num=0))


def problem_493():
    return round(7*(1 - binomial_coef(60, 20)/binomial_coef(70, 20)), 9)


def problem_549(lim=10**8):
    bool_sieve = BoolPrimesSieve(lim)

    def get_smallest_factorial(n):
        best = 0
        for p in primes:
            if n % p != 0:
                continue
            prime_power = 1
            while n % p == 0:
                n //= p
                prime_power *= p

            best = max(best, cache[prime_power])
            if n == 1:
                return best
            if bool_sieve.bool_is_prime(n):
                return max(best, n)
        return best

    cache = dict()
    primes = [x for x in range(2, lim) if bool_sieve.bool_is_prime(x)]

    out = 0
    for i in range(2, lim+1):
        if bool_sieve.bool_is_prime(i):
            power = i
            while power <= lim:
                factorial = result = i
                while factorial % power != 0:
                    result += i
                    factorial *= result
                    factorial %= power
                cache[power] = result
                power *= i
            out += i
        else:
            out += get_smallest_factorial(i)

    return out


def problem_618():
    raise NotImplementedError
    # def fib(k):
    #     a, b = 1, 1
    #     for _ in range(k - 2):
    #         a, b = b, a + b
    #     return b
    #
    # def S(k):
    #     parts = prime_partitions(k)
    #     out_sum = 0
    #     for part in parts:
    #         prod = 1
    #         for num in part:
    #             prod *= num
    #         out_sum += prod
    #     return out_sum
    #
    # return [S(fib(k)) for k in range(2, 12)]


if __name__ == '__main__':
    # print([problem_233(x) for x in range(1, 30)])
    # for n in range(1, 1000000):
    #     if problem_233(n) == 420:
    #         print(n, prime_factors(n))
    lim = 10**8
    print(problem_549())