class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}
        # Iterate through the list
        for i, n in enumerate(nums):
            # Check if the target is >= any number
            # Subtract number from target
            diff = target - n
            # Find an unused number from the list
            if diff in dict:
                # Return matching pair
                return [dict[diff], i]
            else:
                dict[n] = i
        return None