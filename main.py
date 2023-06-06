# This is a sample Python script.
from typing import Optional, List
import math, random, queue


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# 1768
def mergeAlternately(word1: str, word2: str) -> str:
    k = 0
    out = ""
    while k < max(len(word1), len(word2)):
        if k < len(word1):
            out += word1[k]
        if k < len(word2):
            out += word2[k]
        k += 1
    return out


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 2
def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    first = ListNode()
    curNode = first
    e = 0

    while True:
        if not l1:
            l1 = ListNode()
        if not l2:
            l2 = ListNode()
        curVal = l1.val + l2.val + e
        if curVal > 9:
            curVal = curVal - 10
            e = 1
        else:
            e = 0
        curNode.val = curVal
        l1 = l1.next
        l2 = l2.next
        if l1 or l2 or e == 1:
            curNode.next = ListNode()
            curNode = curNode.next
        else:
            return first


def threeSum(nums: List[int]) -> List[List[int]]:
    out = set()
    nums.sort()
    for i in range(len(nums) - 2):
        j, k = i + 1, len(nums) - 1
        while j < k:
            if nums[i] + nums[j] + nums[k] == 0:
                c = (nums[i], nums[j], nums[k])
                out.add(c)
                j += 1
                k -= 1
            if nums[i] + nums[j] + nums[k] < 0:
                j += 1
            if nums[i] + nums[j] + nums[k] > 0:
                k -= 1
    return out


def searchRange(nums: List[int], target: int) -> List[int]:
    # first occurrence
    if len(nums) == 0:
        return [-1, -1]
    lower, upper = 0, len(nums) - 1
    if nums[upper] < target or nums[lower] > target:
        return [-1, -1]
    while nums[upper] > target or (nums[upper] == target and upper > 0 and nums[upper - 1] == target):
        if upper - lower == 1 and not nums[upper] == target:
            if not nums[lower] == target:
                return [-1, -1]
            else:
                return [lower, lower]
        m = math.floor((upper + lower) / 2)
        if nums[m] < target:
            lower = m
        else:
            upper = m

    first = upper
    print(first)
    if first == len(nums) - 1:
        return [first, first]
    lower, upper = first, len(nums) - 1
    while not (nums[lower] == target and (lower == len(nums) - 1 or not nums[lower + 1] == target)):
        m = math.ceil((upper + lower) / 2)
        if nums[m] > target:
            upper = m
        else:
            lower = m
    last = lower
    print(lower)
    return list([first, last])


def search(nums: List[int], target: int) -> int:
    if len(nums) == 0:
        return -1

    def isSorted(nums):
        if len(nums) == 1 or len(nums) == 0:
            return True
        return nums[0] < nums[len(nums) - 1]

    def findPivot(nums: List[int], offset: int):
        if isSorted(nums):
            return 0 + offset
        early = nums[0:math.floor(len(nums) / 2)]
        late = nums[math.floor(len(nums) / 2): len(nums)]
        if isSorted(early):
            if isSorted(late):
                return math.ceil(len(nums) / 2) + offset
            else:
                return findPivot(late, offset)
        else:
            return findPivot(early, offset + math.ceil(len(nums) / 2))

    piv = findPivot(nums, 0)

    def rotRight(k, piv):
        return k - piv if k - piv >= 0 else len(nums) + (k - piv)

    def rotLeft(k, piv):
        return k + piv if k + piv < len(nums) else k + piv - len(nums)

    low = 0
    high = len(nums) - 1
    x = nums[rotRight(low, piv)]
    if nums[rotRight(low, piv)] > target:
        return -1

    while nums[rotRight(low, piv)] < target:
        if low == high:
            return - 1
        m = math.ceil((low + high) / 2)
        n = nums[rotRight(m, piv)]
        if n <= target:
            low = m
        else:
            if m == high and nums[rotRight(m, piv)] != target:
                return -1
            high = m
    y = rotRight(low, piv)
    return y


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    n = len(matrix)
    m = len(matrix[0])

    def ind(k):
        i = math.floor(k / m)
        j = k % m
        return i, j

    low, high = 0, n * m - 1
    while low < high:
        il, jl = ind(low)
        if matrix[il][jl] == target:
            return True
        ih, jh = ind(high)
        if matrix[ih][jh] == target:
            return True
        mid = math.floor((low + high) / 2)
        im, jm = ind(mid)
        if matrix[im][jm] >= target:
            high = mid
        else:
            low = mid
        if high - low == 1 and matrix[im][jm] != target:
            return False
    i, j = ind(low)
    return matrix[i][j] == target


def findMin(nums):
    def isSorted(nums):
        if len(nums) == 1 or len(nums) == 0:
            return True
        return nums[0] < nums[len(nums) - 1]

    def findPivot(nums: List[int], offset: int):
        if isSorted(nums):
            return 0 + offset
        early = nums[0:math.floor(len(nums) / 2)]
        late = nums[math.floor(len(nums) / 2): len(nums)]
        if isSorted(early):
            if isSorted(late):
                return math.ceil(len(nums) / 2) + offset
            else:
                return findPivot(late, offset)
        else:
            return findPivot(early, offset + math.ceil(len(nums) / 2))

    piv = findPivot(nums, 0)

    def rotRight(k, piv):
        return k - piv if k - piv >= 0 else len(nums) + (k - piv)

    def rotLeft(k, piv):
        return k + piv if k + piv < len(nums) else k + piv - len(nums)

    return nums[rotRight(0, piv)]


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    while head and head.val == head.next.val:
        cur = head.next
        while cur.next and cur.val == cur.next.val:
            cur = cur.next
        head = cur.next
    if not head:
        return head
    hold = head
    cur = hold.next
    while cur:
        if cur.next and cur.val == cur.next.val:
            while cur.val == cur.next.val:
                cur = cur.next
            hold.next = cur.next
            cur = hold.next
        else:
            hold = cur
            cur = cur.next
    return head


def backspaceCompare(s: str, t: str) -> bool:
    def reduce(s):
        p = 0
        while p < len(s):
            if s[p] == '#':
                c = p
                while c < len(s) and s[c] == '#':
                    c += 1
                s = s[:(max(0, p + p - c))] + s[c:]
                p = max(-1, 2 * p - c - 1)
            p += 1
        return s

    return reduce(s) == reduce(t)


def maxArea(height):
    i, j = 0, len(height) - 1
    max = 0
    while i < j:
        area = (j - i) * min(height[i], height[j])
        if area > max:
            max = area
        if height[i] < height[j]:
            i = i + 1
        else:
            j = j - 1
    return max


def hardcap(p, hc, n):
    k, w = 0, 0
    while k < n:
        us, them = 0, 0
        weServe = True
        while us < 15 and them < 15:
            if weServe:
                if random.random() > p:
                    us += 1
                else:
                    them += 1
                    weServe = False
            else:
                if random.random() > p:
                    them += 1
                else:
                    us += 1
                    weServe = True
            # print(us, them)
        if us == 15:
            w += 1
        k += 1
        # print(str(w) + " wins in " + str(k) + " attempts")
    return w / k


def intervalIntersection(firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    def intersect(a, b):
        if max(a[0], b[0]) > min(a[1], b[1]):
            return None
        return [max(a[0], b[0]), min(a[1], b[1])]

    if len(firstList) == 0 or len(secondList) == 0:
        return []
    out = []
    a, b = firstList.pop(0), secondList.pop(0)
    if intersect(a, b):
        out.append(intersect(a, b))
    while len(firstList) > 0 or len(secondList) > 0:
        if a[1] > b[1]:
            if secondList:
                b = secondList.pop(0)
            else:
                return out
        elif a[1] < b[1]:
            if firstList:
                a = firstList.pop(0)
            else:
                return out
        else:
            if secondList and firstList:
                a = firstList.pop(0)
                b = secondList.pop(0)
            else:
                return out
        if intersect(a, b):
            out.append(intersect(a, b))
    return out


def firstMissingPositive(nums):
    for k in range(len(nums)):
        c = k + 1
        start = True
        while 0 < c <= len(nums) and nums[c - 1] != c:
            d = nums[c - 1]
            nums[c - 1] = c if not start else -1
            c = d
            start = False
    for k in range(len(nums)):
        if nums[k] == -1:
            return k + 1
    return len(nums) + 1


def findAnagrams(s, p):
    out = []
    target = dict()
    for l in p:
        if l in target:
            target[l] += 1
        else:
            target[l] = 1
    sub = dict()
    for l in s[:len(p)]:
        if l in sub:
            sub[l] += 1
        else:
            sub[l] = 1
    for k in range(len(s) - len(p) + 1):
        eq = True
        for l in target:
            if target[l] != sub[l]:
                eq = False
        if eq:
            out.append(k)
        sub[s[k]] -= 1
        if k + len(p) < len(s):
            if s[k + len(p)] in sub:
                sub[s[k + len(p)]] += 1
            else:
                sub[s[k + len(p)]] = 1
    return out


def subArray(nums, k):
    out = 0
    width = 0
    prod = 1
    for start in range(len(nums)):
        if prod < k:
            while start + width < len(nums) and prod * nums[start + width] < k:
                prod = prod * nums[start + width]
                width += 1
        else:
            while width > 0 and prod >= k:
                width -= 1
                prod = prod / nums[start + width]
        out += width
        if width > 0:
            width -= 1
            prod = prod / nums[start]
    return out


def minSubarray(nums, target):
    c = len(nums) + 1
    s = sum(nums[:len(nums) // 2])
    end = len(nums) // 2
    for start in range(len(nums)):
        low = start
        while not start == end - 1 or not (s >= target > s - nums[end - 1]) or not end == len(nums):
            if s >= target:
                newend = (start + end) // 2
                s = s - sum(nums[newend:end])
                end = newend
            if s < target:
                newend = (end + len(nums)) // 2
                s = s + sum(nums[end:newend])
                end = newend
        if end == len(nums):
            if s >= target:
                c = min(c, end - start)
        c = min(c, end - start)
    return 0 if c == len(nums) + 1 else c


def numIslands(grid: List[List[str]]) -> int:
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "0":
                continue
            else:
                count += 1
                qu = queue.Queue()
                qu.put([i, j])
                grid[i][j] = "0"
                while not qu.empty():
                    c = qu.get()
                    k, l = c[0], c[1]
                    # grid[k][l] = "0"
                    if k + 1 < len(grid) and grid[k + 1][l] == "1":
                        qu.put([k + 1, l])
                        grid[k + 1][l] = "0"
                    if k - 1 > -1 and grid[k - 1][l] == "1":
                        qu.put([k - 1, l])
                        grid[k - 1][l] = "0"
                    if l + 1 < len(grid[k]) and grid[k][l + 1] == "1":
                        qu.put([k, l + 1])
                        grid[k][l + 1] = "0"
                    if l - 1 > -1 and grid[k][l - 1] == "1":
                        qu.put([k, l - 1])
                        grid[k][l - 1] = "0"
    return count


def numberOfProvinces(isConnected):
    seen = {}
    count = 0
    for k in range(len(isConnected)):
        if not k in seen:
            count += 1
            prov = []
            seen[k] = True
            prov.append(k)
            while prov:
                c = prov.pop()
                for i in range(len(isConnected[c])):
                    if isConnected[c][i]:
                        if not i in seen:
                            seen[i] = True
                            prov.append(i)
    return count


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    def isomorphic(a, b):
        if not a:
            return not b
        if not b:
            return not a
        if a.val == b.val:
            return isomorphic(a.left, b.left) and isomorphic(a.right, b.right)
        return False

    def recSub(r, s):
        if isomorphic(r, s):
            return True
        if recSub(r.left, s):
            return True
        if recSub(r, s):
            return True
        else:
            return False

    return recSub(root, subRoot)


def shortestBinaryPath(grid):
    def neighbours(c):
        neighbours = [(c[0] - 1, c[1] - 1), (c[0] - 1, c[1]), (c[0] - 1, c[1] + 1,), (c[0], c[1] - 1), (c[0], c[1] + 1),
                      (c[0] + 1, c[1] - 1), (c[0] + 1, c[1]), (c[0] + 1, c[1] + 1)]
        return neighbours

    visited = {(0, 0): 1}
    q = [(0, 0)]
    while q:
        c = q.pop(0)
        for n in neighbours(c):
            if 0 <= n[0] < len(grid) and 0 <= n[1] < len(grid[n[0]]):
                if grid[n[0]][n[1]] == 0:
                    if n not in visited:
                        visited[n] = visited[c] + 1
                        q.append(n)
                    elif visited[c] + 1 < visited[n]:
                        visited[n] = visited[c] + 1
                        q.append(n)
    if (len(grid) - 1, len(grid) - 1) not in visited:
        return -1
    return visited[(len(grid) - 1, len(grid) - 1)]


def allPaths(graph):
    if graph == [[0]]:
        return graph
    out = []
    q = []
    c = [0]
    q.append(c)
    while q:
        c = q.pop(0)
        if c[-1] == len(graph) - 1:
            out.append(c)
        else:
            for k in graph[c[-1]]:
                n = c.copy()
                n.append(k)
                q.append(n)
    return out


def combSum2(candidates, target):
    out = []
    candidates.sort()

    def rec(curr, candidates, target):
        if target == 0:
            out.append(list(curr))
            return
        for k in candidates:
            if k <= target:
                curr.append(k)
                candidates.remove(k)
                rec(curr, candidates, target - k)
                curr.pop(-1)
                candidates.append(k)

    rec([], candidates, target)
    return out


def minSubArray(target, nums):
    # init slices
    slices = [0]
    for x in nums:
        slices.append(slices[-1] + x)
    # print(slices)
    if slices[-1] < target:
        return 0
    out = len(nums) + 1
    for i in range(len(nums)):  # determine the smallest subarray [i, k] summing to target via bin search
        if slices[-1] - slices[i] < target:
            continue
        low = i + 1
        high = len(nums) + 1
        while slices[low] - slices[i] < target:
            mid = (low + high) // 2
            if slices[mid] - slices[i] < target:
                low = mid + 1
            else:
                high = mid
        out = min(out, low - i)
    return out


def wordBreak(s: str, wordDict: List[str]) -> bool:
    sol = {"": True}

    def recCompute(s):
        if s in sol:
            return sol[s]
        found = False
        for part in wordDict:
            if s[:len(part)] == part:
                recCompute(s[len(part):])
                found = found or sol[s[len(part):]]
        sol[s] = found

    recCompute(s)
    return sol[s]


def maxScore(nums):
    def gcd(a, b):
        if a == 0:
            return b
        return gcd(b % a, a)

    GCD = {}
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            GCD[(i, j)] = gcd(nums[i], nums[j])

    sol = {}

    def recCompute(n, used):
        if n == 0:
            return 0
        if (n, used) in sol:
            return sol[(n, used)]
        else:
            # produce all possible pairs of set bits
            cmax = -1

            for i in range(len(nums)):
                if used & 1 << i:
                    for j in range(i + 1, len(nums)):
                        if used & 1 << j:
                            usedbefore = used & ~(1 << i) & ~(1 << j)
                            cmax = max(cmax, recCompute(n - 1, usedbefore) + n * GCD[(i, j)])
            sol[(n, used)] = cmax
            return cmax

    return recCompute(len(nums) // 2, pow(2, len(nums)) - 1)


# convert integer to binary string
def binary(n):
    if n == 0:
        return "0"
    out = ""
    while n > 0:
        out = str(n % 2) + out
        n = n // 2
    return out


def isValidSudoku(board):
    rows = []
    columns = []
    boxes = []
    for i in range(9):
        for j in range(9):
            if i == 0:
                columns.append(set())
            if j == 0:
                rows.append(set())
            if i % 3 == 0 and j % 3 == 0:
                boxes.append(set())

            if board[i][j] != ".":
                if board[i][j] in columns[j]:
                    return False
                columns[j].add(board[i][j])

                if board[i][j] in rows[i]:
                    return False
                rows[i].add(board[i][j])

                box = 3 * (i // 3) + (j // 3)
                print("box check:")
                print(i, j, box)
                if board[i][j] in boxes[box]:
                    return False
                boxes[box].add(board[i][j])

    return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nums = [3, 4, 6, 8]
    board = [["5", "3", ".", ".", "7", ".", ".", ".", "."]
        , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
        , [".", "9", "8", ".", ".", ".", ".", "6", "."]
        , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
        , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
        , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
        , [".", "6", ".", ".", ".", ".", "2", "8", "."]
        , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
        , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
    print("result " + str(isValidSudoku(board)))
