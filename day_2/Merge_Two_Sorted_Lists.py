# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        l3 = ListNode()
        l4 = l3
       
        # if l1 and l2 exist
        while l1 and l2:
            if l1.val <= l2.val:
                l3.val = l1.val
                l3.next = ListNode()
                l3 = l3.next
                l1 = l1.next
            else:
                l3.val = l2.val
                l3.next = ListNode()
                l3 = l3.next
                l2 = l2.next
        
        # if only l1 exist
        if l1:
            while l1:
                l3.val = l1.val
                l1 = l1.next
                # keep adding nodes to l3 only if l1 is not empty at this point
                if l1:
                    l3.next = ListNode()
                    l3 = l3.next

        # if only l2 exist    
        elif l2:
            while l2:
                l3.val = l2.val
                l2 = l2.next
                # keep adding nodes to l3 only if l2 is not empty at this point
                if l2:
                    l3.next = ListNode()
                    l3 = l3.next

        # if both lists are empty
        else:
            return None
        
        return l4


# if (not l1):
#     return l2
# elif (not l2):
#     return l1
# else:
#     if l1.val < l2.val:
#         l = ListNode(l1.val)
#         l.next = self.mergeTwoLists(l1.next, l2)
#         return l
#     else:
#         l = ListNode(l2.val)
#         l.next = self.mergeTwoLists(l2.next, l1)
#         return l