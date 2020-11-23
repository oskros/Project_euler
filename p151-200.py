from helper import *


def problem_151(monte_carlo=False):
    if monte_carlo:
        simulations = 10**9
        def cut_paper(cur_folder):
            draw = random.sample(cur_folder, 1)[0]
            cur_folder.remove(draw)
            if draw == 1:
                cur_folder.extend([2, 3, 4, 5])
            elif draw == 2:
                cur_folder.extend([3, 4, 5])
            elif draw == 3:
                cur_folder.extend([4, 5])
            elif draw == 4:
                cur_folder.append(5)

        prob = 0
        for i in range(1, simulations+1):
            folder = [2, 3, 4, 5]
            count = 0
            for _ in range(14):
                if len(folder) == 1:
                    count += 1
                cut_paper(folder)
            prob = prob + (count - prob)/i
        return prob
    else:
        def evaluate(sheets):
            num_sheets = sum(sheets)

            single = 0
            if num_sheets == 1 == sheets[-1]:
                return 0
            if num_sheets == 1 and sheets[0] == 0:
                single = 1

            for i in range(len(sheets)):
                if sheets[i] == 0:
                    continue
                nxt = sheets.copy()
                nxt[i] -= 1
                for j in range(i + 1, len(nxt)):
                    nxt[j] += 1
                prob = sheets[i] / num_sheets
                single += evaluate(nxt) * prob
            return single

        sheets = [1, 0, 0, 0, 0]
        return evaluate(sheets)


def problem_162():
    @Memoize
    def count(digits, have_other=False, have_zero=False, have_one=False, have_a=False):
        if have_zero and have_one and have_a and digits < 15:
            return 16**digits
        if digits == 0:
            return 0

        nxt = count(digits - 1, True, have_zero, have_one, have_a)
        res = 13 * nxt

        res += nxt if have_zero else count(digits-1, have_other, have_other, have_one, have_a)
        res += nxt if have_one else count(digits-1, True, have_zero, True, have_a)
        res += nxt if have_a else count(digits-1, True, have_zero, have_one, True)

        return res
    return hex(count(16)).upper()[2:]


def problem_164():
    @Memoize
    def create_num(a, b, size):
        if size == 0:
            return 1

        tot = 0
        for n in range(10):
            if size == 20 and n == 0:
                continue
            if a + b + n > 9:
                break
            tot += create_num(b, n, size-1)
        return tot
    return create_num(0, 0, 20)


def problem_169(n=10**25, naive=False):
    def naive_counting(lim):
        def count_with_memory(num, pows, used):
            if num == 0:
                return 1

            count = 0
            for p in pows:
                if used.count(p) >= 2:
                    continue
                new_pows = tuple(x for x in pows if x <= p)
                count += count_with_memory(num - p, new_pows, used + (p,))
            return count

        exp_bound = int(math.log(max(lim, 1), 2))
        powers = tuple(2 ** x for x in range(exp_bound + 1))
        return count_with_memory(lim, powers, ())

    @Memoize
    def count_ways(num):
        if num == 0:
            return 1
        if num % 2 == 0:
            return count_ways(num // 2) + count_ways((num - 2) // 2)
        return count_ways((num - 1) // 2)

    if naive:
        return naive_counting(lim=n)
    return count_ways(n)


def problem_179(lim=10**7):
    arr = np.array([1]*lim)
    for i in range(1, lim):
        arr[i::i+1] += 1
    return np.sum([x == 0 for x in np.diff(arr)])


def problem_183():
    def D(N):
        k = round(N / math.e)
        while k % 2 == 0:
            k /= 2
        while k % 5 == 0:
            k /= 5
        if N % k == 0:
            return -N
        else:
            return N

    out = 0
    for N in range(5, 10001):
        out += D(N)
    return out


def problem_187(lim=10**8):
    sieve = primes_sieve(lim//2)
    count = 0
    lb = -1
    for p in sieve:
        lb += 1
        ub = bisect.bisect_right(sieve, lim // p)
        if lb > ub:
            break
        count += ub - lb
    return count


def problem_188():
    a = 1777
    b = 1855
    modulo = 10**8
    return tetration(a, b, modulo)


def problem_191():
    # This method works, but is much slower...
    """
    global count
    count = 0

    def make_str(prev):
        if len(prev) == 30:
            global count
            count += 1
            return
        for s in ['A', 'L', 'O']:
            if s == 'L' and prev.count(s) > 0:
                continue
            if s == 'A' and prev[-2:].count(s) == 2:
                continue
            make_str(prev + [s])
    make_str([])
    """

    @Memoize
    def count_str(absence, late, size):
        if absence > 2 or late > 1:
            return 0
        if size == 0:
            return 1

        tot = 0
        tot += count_str(absence=0, late=late, size=size-1)
        tot += count_str(absence=absence+1, late=late, size=size-1)
        tot += count_str(absence=0, late=late+1, size=size-1)

        return tot

    return count_str(0, 0, 30)


if __name__ == '__main__':
    from pprint import pprint
    pprint(problem_183())
