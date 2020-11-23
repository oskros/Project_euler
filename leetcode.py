def two_sum(nums, target):
    for i, n1 in enumerate(nums):
        other = target - n1
        for j, n2 in enumerate(nums[i+1:], 1):
            if other == n2:
                return i, i + j


def two_sum2(nums, target):
    for i, n in enumerate(nums):
        t = target - n
        if t in nums[i+1:]:
            return i, i+1+nums[i+1:].index(t)


if __name__ == '__main__':
    print(two_sum2([2, 7, 11, 15], 9))
    print(two_sum2([3,3], 6))