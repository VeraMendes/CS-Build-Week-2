class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        
        # create a set to save previous numbers checked
        prev_numbers = set()
        
        # iterate through the array, until you find one dup
        for num in nums:
            
            # if dup found, return True
            if num in prev_numbers:
                return True
            # add numb to previous numbers set
            else:
                prev_numbers.add(num)
                
        # in the end, if no dupes found, return False
        return False