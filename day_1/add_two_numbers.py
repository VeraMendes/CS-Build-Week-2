# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

        
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        def get_list_number(linked_list):
            decimals = 1
            list_result_number = 0
            while linked_list:
                list_result += linked_list.val * decimals
                linked_list = linked_list.next
                decimals *= 10
            return list_result_number
        
        l1_result = get_list_number(l1)
        l2_result = get_list_number(l2)

        l3_result = l1_result + l2_result
        
        l3 = ListNode()
        # saving a copy of l3 values, as l3 will be moving next
        l4 = l3
        # invert result
        reverse_string = str(l3_result)[::-1]
        print(reverse_string)
        # split
        l3_array = list(reverse_string)
        # insert every value on split into the new linked list
        for i, num in enumerate(l3_array):
            l3.val = num
            if i < len(l3_array)-1:
                l3.next = ListNode()
                l3 = l3.next
            
        return l4