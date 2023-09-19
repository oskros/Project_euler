from helper import *


def problem_51():
    primes = [str(x) for x in primes_sieve(10**6) if x > 10**5]
    for p in primes:
        dig = next((x for x in '012' if p.count(x) in (3, 6, 9)), None)
        if dig is None:
            continue

        composites = 0
        for i in range(int(dig)+1, 10):
            if not miller_rabin(int(p.replace(dig, str(i)))):
                composites += 1
            if composites > 1:
                break
        else:
            return int(p)


def problem_52():
    return next(i for i in range(10**5, 10**7) if all(set(str(i)) == set(str(i*x)) for x in range(2, 7)))


def problem_53():
    return sum(1 for n in range(1, 101) for r in range(1, n+1) if binomial_coef(n, r) > 10**6)


def problem_55():
    def is_lychrel(n):
        for j in range(50):
            n += int(str(n)[::-1])
            if str(n) == str(n)[::-1]:
                return False
        return True

    return sum(1 for n in range(10000) if is_lychrel(n))


def problem_56():
    return max([digit_sum(a**b) for a in range(100) for b in range(100)])


def problem_57():
    out = 0
    num, denom = 3, 2
    for _ in range(1000):
        if len(str(num)) > len(str(denom)):
            out += 1
        num, denom = 2*denom+num, num+denom
    return out


def problem_58():
    diagonal_number, side_len, diagonal_count, prime_count = 1, 1, 1, 0

    prime_ratio = 1
    while prime_ratio > 0.1:
        side_len += 2
        for _ in range(4):
            diagonal_number += side_len - 1
            diagonal_count += 1
            if is_prime_naive(diagonal_number):
                prime_count += 1
        prime_ratio = prime_count/diagonal_count

    print(prime_ratio, side_len)


def problem_60():
    plist = primes_sieve(10000)
    cl = lambda x, y: [int(str(x) + str(y)), int(str(y) + str(x))]

    for ix, x in enumerate(plist):
        for iy, y in enumerate(plist[ix:]):
            if all(miller_rabin(x) for x in cl(x, y)):
                for iw, w in enumerate(plist[ix+iy:]):
                    if all(miller_rabin(x) for x in cl(x, w) + cl(y, w)):
                        for iz, z in enumerate(plist[ix+iy+iw:]):
                            if all(miller_rabin(x) for x in cl(x, z) + cl(y, z) + cl(w, z)):
                                for k in plist[ix+iy+iw+iz:]:
                                    if all(miller_rabin(x) for x in cl(x, k) + cl(y, k) + cl(z, k) + cl(w, k)):
                                        return x+y+z+w+k


def problem_61():
    def polyg_4digit(degree):
        if degree == 3:
            rng = range(45, 141)
        elif degree == 4:
            rng = range(32, 100)
        elif degree == 5:
            rng = range(26, 82)
        elif degree == 6:
            rng = range(23, 71)
        elif degree == 7:
            rng = range(21, 64)
        elif degree == 8:
            rng = range(20, 59)
        else:
            raise NotImplementedError
        return [polygonal(degree, x) for x in rng]

    def concat_4digit(n1, n2):
        return int(str(n1)[2:] + str(n2)[:2])

    k3 = polyg_4digit(3)
    k4 = polyg_4digit(4)
    k5 = polyg_4digit(5)
    k6 = polyg_4digit(6)
    k7 = polyg_4digit(7)
    k8 = polyg_4digit(8)

    tmp_map = {3: k3, 4: k4, 5: k5, 6: k6, 7: k7, 8: k8}

    def temp_func(loopers, checkers):
        p3, p4, p5 = loopers
        p6, p7, p8 = checkers
        for i3 in p3:
            for i4 in p4:
                for i5 in p5:
                    i3i4 = concat_4digit(i3, i4)
                    i4i5 = concat_4digit(i4, i5)
                    i5i3 = concat_4digit(i5, i3)

                    if i3i4 in p6 and i4i5 in p7 and i5i3 in p8:
                        return i3, i3i4, i4, i4i5, i5, i5i3
                    if i3i4 in p6 and i4i5 in p8 and i5i3 in p7:
                        return i3, i3i4, i4, i4i5, i5, i5i3
                    if i3i4 in p7 and i4i5 in p6 and i5i3 in p8:
                        return i3, i3i4, i4, i4i5, i5, i5i3
                    if i3i4 in p7 and i4i5 in p8 and i5i3 in p6:
                        return i3, i3i4, i4, i4i5, i5, i5i3
                    if i3i4 in p8 and i4i5 in p6 and i5i3 in p7:
                        return i3, i3i4, i4, i4i5, i5, i5i3
                    if i3i4 in p8 and i4i5 in p7 and i5i3 in p6:
                        return i3, i3i4, i4, i4i5, i5, i5i3

    permutations = [(x, tuple(set(tmp_map.keys()) - set(x))) for x in list(itertools.permutations(tmp_map.keys(), 3))]
    for permutation in permutations:
        loopers = [tmp_map[x] for x in permutation[0]]
        checkers = [tmp_map[x] for x in permutation[1]]
        out = temp_func(loopers=loopers, checkers=checkers)
        if out is not None:
            return sum(out), out


def problem_62():
    cube_len = 8
    while True:
        cubes = cubes_with_len(cube_len)

        for ix, x in enumerate(cubes):
            count = 0
            for y in cubes[ix:]:
                if contains_same_digits(x, y):
                    count += 1
                if count == 5:
                    return x
        cube_len += 1


def problem_63():
    count = 0
    for power in range(1, 100):
        s = 1
        while len(str(s ** power)) <= power:
            if len(str(s ** power)) == power:
                count += 1
            s += 1
    return count


def problem_64():
    count = 0
    for n in range(2, 10000):
        sq_n = n**0.5
        if int(sq_n) == sq_n:
            continue
        period_count = 0

        m, d, a = 0, 1, n ** 0.5 // 1
        known = []
        while True:
            m = d * a - m
            d = (n - m ** 2) / d
            a = ((n ** 0.5 + m) / d) // 1
            if (d, m) in known:
                break
            period_count += 1
            known.append((d, m))
        if period_count % 2 != 0:
            count += 1
    return count


def problem_65():
    e = [2, 1, 2]

    for i in range(4, 70, 2):
        e.extend((1, 1, i))

    ie = iter(e)
    r, s = 0, 1
    for i in range(1, 101):
        a = next(ie)
        r, s = s, a * s + r

    out = 0
    for n in str(s):
        out += int(n)
    return out


def problem_66():
    out = {}
    for D in range(2, 1001):
        sq_D = D**0.5
        if int(sq_D) == sq_D:
            continue

        m, d, a = 0, 1, int(sq_D)
        x, x_1 = a, 1
        y, y_1 = 1, 0

        while True:
            if x * x - D * y * y == 1:
                out[D] = int(x)
                break

            m = d * a - m
            d = int((D - m * m) / d)
            a = int((sq_D + m) / d)

            x_2, x_1 = x_1, x
            y_2, y_1 = y_1, y

            x = a*x_1 + x_2
            y = a*y_1 + y_2
    return max(out, key=out.get)


def problem_67():
    triangle = []
    with open('files/p067_triangle.txt') as fo:
        for line in fo:
            triangle.append([int(x) for x in line.split()])
    n_rows = len(triangle)
    for i in range(1, n_rows):
        cur_row = triangle[n_rows - 1 - i]
        next_row = triangle[n_rows - i]

        for j in range(len(cur_row)):
            sum_1 = cur_row[j] + next_row[j]
            sum_2 = cur_row[j] + next_row[j + 1]
            cur_row[j] = max(sum_1, sum_2)
    return triangle[0]


def problem_68():
    perms = list(itertools.permutations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    out = set()
    for p0 in perms:
        v0 = (p0[0], p0[1], p0[2])
        v1 = (p0[3], p0[2], p0[4])
        v2 = (p0[5], p0[4], p0[6])
        v3 = (p0[7], p0[6], p0[8])
        v4 = (p0[9], p0[8], p0[1])

        if not sum(v0) == sum(v1) == sum(v2) == sum(v3) == sum(v4):
            continue
        v = [v0, v1, v2, v3, v4]
        min_vec = np.argmin([x[0] for x in v])
        vec_srt = v[min_vec:] + v[:min_vec]
        conc_vec = sum(vec_srt, ())

        final_number = concat_list(conc_vec)
        if len(str(final_number)) == 16:
            out.add(final_number)
    return max(out)


def problem_69():
    s = 1
    prime_list = primes_sieve(100)
    i = 0
    while s * prime_list[i] < 1000000:
        s *= prime_list[i]
        i += 1
    return s


def problem_70():
    totient_list = totient_sieve(10000000)
    permutations_list = []
    for idx, val in enumerate(totient_list, 1):
        if contains_same_digits(idx, val):
            permutations_list.append((idx, idx/val))
    return min(permutations_list, key=lambda x: x[1])


def problem_71():
    max_no = int(1e6)
    target = 3/7
    n, d = 1, 1
    closest = 999
    closest_pair = (0, 0)
    count = 0
    while d <= max_no:
        count += 1
        diff = target - n / d
        if 0 < diff < closest:
            closest = diff
            closest_pair = (n, d)
        if n / d < target:
            n += 1
        else:
            d += 1
    return count, closest_pair


def problem_72():
    return sum(totient_sieve(1000000)) - 1


def problem_73():
    max_no = 12000
    out = []
    for i in range(2, max_no + 1):
        for j in range(1, i):
            if gcd(i, j) == 1:
                out += [(j, i)]
    srt = sorted(out, key=lambda x: x[0]/x[1])
    count = sum([1 for x in srt if 1/3 < x[0]/x[1] < 1/2])
    return count


def problem_74():
    count = 0
    for x in range(1, 1000000):
        tmp = len_of_factorial_chain(x)
        if tmp == 60:
            count += 1
    return count


def problem_75():
    limit = 1500000
    triangles = [0] * (limit + 1)

    result = 0
    mlimit = int((limit / 2) ** 0.5)

    for m in range(2, mlimit):
        for n in range(1, m):
            if (n + m) % 2 == 1 and gcd(n, m) == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                p = a + b + c
                while p <= limit:
                    triangles[p] += 1
                    if triangles[p] == 1: result += 1
                    if triangles[p] == 2: result -= 1
                    p += a + b + c
    return result


def problem_76():
    n = 100
    parts = [1]
    all_polygs = pentag_range_below(n)
    i = 1
    while True:
        tmp_polygs = [x for x in all_polygs if x <= i]
        signs = ([1, 1, -1, -1] * (len(tmp_polygs) // 4 + 1))[:len(tmp_polygs)]
        parts.append(sum([y*parts[i-x] for x, y in zip(tmp_polygs, signs)]))

        if i == n:
            return parts[i]-1
        i += 1


def problem_77():
    max_n = 5000

    # We guess a bound on the number. As 100 generated an awful lot of partitions in problem 76, we use this as a bound
    guess_bound = 100

    # Generate a list of primes up to the guesstimated bound
    primes = primes_sieve(guess_bound)

    # Initialize list to hold number of prime partitions for each index, initializing with ways[0] = 1
    ways = [0] * (guess_bound + 1)
    ways[0] = 1

    # Use dynamic programming to solve sub-problems as follows:
    # For the first prime, add a partition count for each number it can generate below the bound. As 0 can be generated
    # once, we find ways[2] = 1, ways[3] = 0, ways[4] = 1, ways[5] = 0, ways[6] = 1, ways[7] = 0, ways[8] = 1, ...
    # For the second prime, we follow the same pattern:
    # ways[3] = 1, ways[4] = 1, ways[5] = 1, ways[6] = 2, ways[7] = 1, ways[8] = 2 ...
    # For the third prime, we get:
    # ways[5] = 2, ways[6] = 2, ways[7] = 2, ways[8] = 3, ...
    for prime in primes:
        for j in range(prime, guess_bound + 1):
            ways[j] += ways[j - prime]

    # After iterating through all primes below our guesstimated bound, we should have created all partitions possible
    # If one partition is greater than 5000 (asked by Project Euler), we return it - if not, the iterator below raises
    # an error (and the guesstimate should be increased)
    return next(i for i, elem in enumerate(ways) if elem > max_n)


def problem_78():
    parts = [1]
    all_polygs = pentag_range_below(100000)
    i = 1
    while True:
        tmp_polygs = [x for x in all_polygs if x <= i]
        signs = ([1, 1, -1, -1] * (len(tmp_polygs) // 4 + 1))[:len(tmp_polygs)]
        parts.append(sum([y*parts[i-x] for x, y in zip(tmp_polygs, signs)]))

        if parts[i] % 1000000 == 0:
            return i
        i += 1


def problem_79():
    with open('files/p079_keylog.txt', 'r') as fo:
        p = fo.readlines()
    p = list(set([x.strip() for x in p]))
    before_cipher = {}
    after_cipher = {}
    for elem in p:
        before_cipher.setdefault(elem[2], []).extend([elem[0], elem[1]])
        before_cipher.setdefault(elem[1], []).append(elem[0])
        after_cipher.setdefault(elem[0], []).extend([elem[1], elem[2]])
        after_cipher.setdefault(elem[1], []).append(elem[2])
    before_cipher = {key: sorted(list(set(value))) for key, value in before_cipher.items()}
    after_cipher = {key: sorted(list(set(value))) for key, value in after_cipher.items()}

    for i in range(10):
        if str(i) not in before_cipher.keys() and str(i) in after_cipher.keys():
            before_cipher[str(i)] = []
        if str(i) not in after_cipher.keys() and str(i) in before_cipher.keys():
            after_cipher[str(i)] = []
    correct_order = sorted(after_cipher, key=lambda x: len(after_cipher[x]), reverse=True)
    return int(''.join(correct_order))


def problem_80():
    return sum([sum(map(int, str(int_sqrt(x*10**200))[:100])) for x in range(2, 101) if int(x**0.5) != x**0.5])


def problem_81():
    m = [[int(x) for x in line.strip().split(',')] for line in open('files/p081_matrix.txt', 'r')]
    dim = len(m)

    for row in range(dim):
        for col in range(dim):
            up = float('inf')
            left = float('inf')
            if col > 0:
                left = m[row][col - 1]
            if row > 0:
                up = m[row - 1][col]
            if row or col:
                m[row][col] += min(left, up)

    return m[-1][-1]


def problem_82():
    m = [[int(x) for x in line.strip().split(',')] for line in open('files/p081_matrix.txt', 'r')]
    dim = len(m)

    # Get second to last column
    out = [0]*dim
    for i in range(dim):
        out[i] = m[i][dim - 1]

    # Loop backwards starting with second to last column (last column is dim-1, second to last is dim-2)
    # We utilise dynamic programming by solving each column separately going backwards through the columns
    for i in range(dim-2, -1, -1):
        # For the first row we always go right
        out[0] += m[0][i]
        # Loop through the rest of the rows, checking if going up or going right is cheaper
        for j in range(1, dim):
            move_up = out[j-1] + m[j][i]
            move_right = out[j] + m[j][i]
            out[j] = min(move_up, move_right)

        # Loop backwards through the rows, this time checking if going down and right are better than previous solution
        for j in range(dim-2, -1, -1):
            previous_solution = out[j]
            down_and_right = out[j+1] + m[j][i]
            out[j] = min(previous_solution, down_and_right)
    return min(out)


def problem_83():
    m = [[int(x) for x in line.strip().split(',')] for line in open('files/p081_matrix.txt', 'r')]
    return AStarSearch(matrix=m).result


def problem_84():
    monopoly_odds = MonopolyOdds()
    simulation = monopoly_odds.run(iterations=1000000)

    common_3_tiles = sorted(list(enumerate(simulation)), key=lambda x: x[1], reverse=True)[:3]
    modal_string = ''.join([str(x[0]) if len(str(x[0])) > 1 else '0'+str(x[0]) for x in common_3_tiles])

    return modal_string


def problem_85():
    max_count = 2e6

    closest = 0
    i, j = 1, 1
    out = (i, j)

    while rectangle_count(1, j - 1) < max_count:
        grid = rectangle_count(i, j)
        if abs(max_count - grid) < abs(max_count - closest):
            closest = grid
            out = (i, j)
        if grid > max_count:
            i = 0
            j += 1
        i += 1
    return out[0]*out[1]


def problem_86():
    target = 1e6
    count = 0
    l = 2
    while count < target:
        l += 1
        for wh in range(3, 2*l + 1):
            sp = (l * l + wh * wh)**0.5
            if sp == int(sp):
                if wh <= l:
                    count += wh // 2
                else:
                    count += 1 + l - math.ceil(wh/2)
    return l, count


def problem_87():
    max_no = 5e7
    plist = primes_sieve(int(max_no ** 0.5))
    min_cube = plist[0]**3
    min_quad = plist[0]**4

    out = set()
    for i in plist:
        sq = i ** 2
        if sq + min_cube + min_quad > max_no:
            break
        for j in plist:
            cub = j ** 3
            if cub + sq + min_quad > max_no:
                break
            for k in plist:
                quad = k ** 4
                tmp_sum = quad + cub + sq
                if tmp_sum > max_no:
                    break
                out.add(tmp_sum)
    return len(out)


def problem_88():
    def minimal_product_sum(k):
        start = k
        while True:
            start += 1
            pf = prime_factors(start)
            padded_sum = sum(pf) + k - len(pf)

            if padded_sum == start:
                return start
            elif padded_sum < start:
                mp = multiplicative_partitions(start)
                for part in mp:
                    if sum(part) + k - len(part) == start:
                        return start

    tmp = set()
    for x in range(2, 12001):
        mps = minimal_product_sum(x)
        tmp.add(mps)
    return sum(tmp)


def problem_89():
    roman_txt = [line.strip() for line in open("files/p089_roman.txt", 'r')]
    saved_chars = 0
    for num in roman_txt:
        saved_chars += len(num) - len(integer_to_roman(roman_to_integer(num)))

    return saved_chars


def problem_90():
    def valid_face_comb(lst_a, lst_b):
        lst_a0, lst_a1, lst_a2, lst_a3, lst_a4, lst_a5, lst_a6, lst_a8 = \
            0 in lst_a, 1 in lst_a, 2 in lst_a, 3 in lst_a, 4 in lst_a, 5 in lst_a, 6 in lst_a or 9 in lst_a, 8 in lst_a
        lst_b0, lst_b1, lst_b2, lst_b3, lst_b4, lst_b5, lst_b6, lst_b8 = \
            0 in lst_b, 1 in lst_b, 2 in lst_b, 3 in lst_b, 4 in lst_b, 5 in lst_b, 6 in lst_b or 9 in lst_b, 8 in lst_b

        # Check 01
        if (lst_a0 and lst_b1) or (lst_a1 and lst_b0):
            # Check 04
            if (lst_a0 and lst_b4) or (lst_a4 and lst_b0):
                # Check 06 (aka 09)
                if (lst_a0 and lst_b6) or (lst_a6 and lst_b0):
                    # Check 16
                    if (lst_a1 and lst_b6) or (lst_a6 and lst_b1):
                        # Check 18 (aka 81)
                        if (lst_a1 and lst_b8) or (lst_a8 and lst_b1):
                            # Check 25
                            if (lst_a2 and lst_b5) or (lst_a5 and lst_b2):
                                # Check 36
                                if (lst_a3 and lst_b6) or (lst_a6 and lst_b3):
                                    # Check 46 (aka 64)
                                    if (lst_a4 and lst_b6) or (lst_a6 and lst_b4):
                                        return True
        return False

    count = 0
    for a1 in range(0, 5):
        for a2 in range(a1 + 1, 6):
            for a3 in range(a2 + 1, 7):
                for a4 in range(a3 + 1, 8):
                    for a5 in range(a4 + 1, 9):
                        for a6 in range(a5 + 1, 10):
                            dice1 = [a1, a2, a3, a4, a5, a6]
                            for b1 in range(0, 5):
                                for b2 in range(b1 + 1, 6):
                                    for b3 in range(b2 + 1, 7):
                                        for b4 in range(b3 + 1, 8):
                                            for b5 in range(b4 + 1, 9):
                                                for b6 in range(b5 + 1, 10):
                                                    dice2 = [b1, b2, b3, b4, b5, b6]
                                                    if valid_face_comb(dice1, dice2):
                                                        count += 1
    # We will count double due to the fact that face1 = x, face2 = y is the same as face1 = y, face2 = x. Thus we divide
    # by 2 here.
    return count//2


def problem_91():
    def is_right_angled(P, Q):
        def euclidean_norm(A, B=None):
            if B is None:
                B = (0, 0)
            return (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2

        s1 = euclidean_norm(P, Q)
        s2 = euclidean_norm(Q)
        s3 = euclidean_norm(P)

        sides = sorted([s1, s2, s3])
        if sum(sides[:2]) == sides[2]:
            return True
        return False

    count = 0
    for p1 in range(51):
        for p2 in range(51):
            if (p1, p2) == (0, 0):
                continue
            for q1 in range(51):
                for q2 in range(51):
                    if (q1, q2) == (0, 0) or (q1, q2) == (p1, p2):
                        continue
                    if is_right_angled(P=(p1, p2), Q=(q1, q2)):
                        count += 1
    return count//2


def problem_92():
    lim = 1e7
    arrives_at_89 = defaultdict(lambda: None)
    arrives_at_89[89] = 1
    arrives_at_89[1] = 0

    for i in range(1, int(lim)):
        sq_digit = i
        tmp_map = []
        while True:
            tmp_map.append(sq_digit)
            if arrives_at_89[sq_digit] == 1:
                for num in tmp_map:
                    arrives_at_89[num] = 1
                break
            if arrives_at_89[sq_digit] == 0:
                for num in tmp_map:
                    arrives_at_89[num] = 0
                break
            sq_digit = sum(int(x)**2 for x in str(sq_digit))
    return sum([value for key, value in arrives_at_89.items() if key < lim])


def problem_93():
    def integer_targets_with_arithmetics(a, b, c, d):
        out = set()

        number_perms = list(itertools.permutations([a, b, c, d]))
        op_perms = list(itertools.product(['+', '-', '*', '/'], repeat=3))
        for num_perm in number_perms:
            for op_perm in op_perms:
                op1, op2, op3 = op_perm
                a, b, c, d = num_perm
                res1 = '((%s %s %s) %s %s) %s %s' % (a, op1, b, op2, c, op3, d)
                res2 = '(%s %s %s) %s (%s %s %s)' % (a, op1, b, op2, c, op3, d)
                res3 = '(%s %s (%s %s %s)) %s %s' % (a, op1, b, op2, c, op3, d)
                res4 = '%s %s ((%s %s %s) %s %s)' % (a, op1, b, op2, c, op3, d)
                res5 = '%s %s (%s %s (%s %s %s))' % (a, op1, b, op2, c, op3, d)

                results = [res1, res2, res3, res4, res5]
                for res in results:
                    try:
                        val = eval(res)
                        if val == int(val) and val > 0:
                            out.add(int(val))
                    except ZeroDivisionError:
                        continue
        return sorted(list(out))

    max_nums = None
    longest = 0
    numbers_to_run = itertools.combinations([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    for num in numbers_to_run:
        conseq = conseq_integers(integer_targets_with_arithmetics(*num))

        if conseq > longest:
            longest = conseq
            max_nums = num
    return longest, max_nums


def problem_94():
    x = 2
    y = 1
    limit = 1e9

    result = 0
    while True:
        a3 = 2 * x - 1
        area3 = y * (x - 2)
        if a3 > limit:
            break

        if a3 > 0 and area3 > 0 and a3 % 3 == 0 and area3 % 3 == 0:
            result += a3 + 1

        a3 = 2 * x + 1
        area3 = y * (x + 2)
        if a3 > 0 and area3 > 0 and a3 % 3 == 0 and area3 % 3 == 0:
            result += a3 - 1

        next_x = 2 * x + y * 3
        next_y = y * 2 + x

        x = next_x
        y = next_y
    return result


def problem_95():
    max_sod = 1e6
    longest_chain = 0
    smallest_number = float('Inf')

    amicable = defaultdict(lambda: None)
    amicable[1] = False
    for i in range(1, int(max_sod) + 1):
        sod = sum_of_divisors(i)
        if amicable[i] is not None:
            continue
        if sod == i:
            amicable[i] = False
            continue
        chain = [i]

        while True:
            chain.append(sod)
            if amicable[sod] is False or sod > max_sod:
                for elem in chain:
                    amicable[elem] = False
                break
            next_sod = sum_of_divisors(sod)
            if next_sod in chain and amicable[sod] is None:
                for elem in chain:
                    amicable[elem] = True
                conc_chain = chain[chain.index(next_sod):]
                chain_len = len(conc_chain)
                if chain_len == longest_chain:
                    smallest_number = min(smallest_number, min(conc_chain))
                elif chain_len > longest_chain:
                    longest_chain = chain_len
                    smallest_number = min(conc_chain)
                break
            sod = next_sod
            if sod in chain:
                break
    return smallest_number


def problem_96():
    with open('files/p096_sudoku.txt', 'r') as fo:
        out = []
        tmp = []
        for line in fo:
            if 'Grid' in line:
                if tmp:
                    out.append(tmp)
                tmp = []
            else:
                tmp2 = []
                for no in line.strip():
                    tmp2.append(int(no))
                tmp.append(tmp2)
        out.append(tmp)

    final_sum = 0
    for sudoku in out:
        solved = SudokuSolver(sudoku).grid
        final_sum += concat_list(solved[0][:3])
    return final_sum


def problem_97():
    power = 7830457
    factor = 28433
    addition = 1

    for i in range(1, power + 1):
        factor *= 2
        factor %= 10**10
    factor += addition

    return factor


def problem_98():
    with open('files/p098_words.txt', 'r') as fo:
        raw_file = list(eval(fo.read()))

    anagrams = []
    for i in range(len(raw_file)):
        s_word = ''.join(sorted(raw_file[i]))
        for j in range(i + 1, len(raw_file)):
            if s_word == ''.join(sorted(raw_file[j])):
                anagrams.append((raw_file[i], raw_file[j]))
    max_anagram = len(max(anagrams, key=lambda x: len(x[0]))[0])

    squares = {}
    for n in range(2, max_anagram + 1):
        lower = int(math.ceil((10**(n-1))**0.5))
        upper = int(math.ceil((10**n)**0.5))
        squares[n] = [x * x for x in range(lower, upper)]

    out = 0
    for word1, word2 in anagrams:
        for square in squares[len(word1)]:
            mapping = word_number_mapping(word1, square)
            if mapping is not None:
                anagram_int = int(''.join([mapping[char] for char in word2]))
                if anagram_int in squares[len(word1)]:
                    out = max(out, square, anagram_int)

    return out


def problem_99():
    with open('files/p099_base_exp.txt', 'r') as fo:
        temp = 1
        j = 0
        for line in fo:
            j = j + 1
            x, y = map(int, line.split(','))
            bar = y * math.log(x)
            if bar > temp:
                stat = j
                temp = bar
    return stat


def problem_100():
    P, Q, K, R, S, L = 3, 2, -2, 4, 3, -3
    x, n = 1, 1

    while n < 1e12:
        x, n = P*x + Q*n + K, R*x + S*n + L

    return x, n


if __name__ == '__main__':
    print(problem_85())
