"""
COMS W4701 Artificial Intelligence - Homework 0

In this assignment you will implement a few simple functions reviewing
basic Python operations and data structures.

@author: XINYUE TAN (xt2215)
"""


def manip_list(list1, list2):
    # Print the last element of list1.
    print("the last element of list1 is: ", list1[len(list1) - 1])
    # Remove the last element of list1.
    del list1[len(list1) - 1]
    # Change the second element of list2 to be identical to the first element of list1.
    list2[1] = list1[0]
    # Print a concatenation of list1 and list2 without modifying the two lists.
    print("the concatenation of list1 and list2 is: ", list1 + list2)
    # Return a single list consisting of list1 and list2 as its two elements.
    return [list1, list2]  # replace this


def manip_tuple(obj1, obj2):
    # Create a tuple of the two object parameters.
    t = (obj1, obj2)
    # Attempt to modify the tuple by reassigning the first item--Python should throw an exception upon execution.
    t[0] = 1
    return None


def manip_set(list1, list2, obj):
    # Create a set called set1 using list1.
    set1 = set(list1)
    # Create a set called set2 using list2.
    set2 = set(list2)
    # Add obj to set1.
    set1.add(obj)
    # Test if obj is in set2 (print True or False).
    print("if ", obj, " is in set2: ", obj in set2)
    # Print the difference of set1 and set2.
    # not explicit on whether it is s1-s2 or s2-s1
    # so I print the symmetric difference
    # print(set1.difference(set2).union(set2.difference(set1)))
    print("the symmetric difference of set1 and set2 is: ", (set1 - set2) | (set2 - set1))
    # Print the union of set1 and set2.
    # print(set1.union(set2))
    print("the union of set1 and set2 is: ", set1 | set2)
    # Print the intersection of set1 and set2.
    # print(set1.intersection(set2))
    print("the intersection of set1 and set2 is: ", set1 & set2)
    # Remove obj from set1.
    set1.remove(obj)
    return None


def manip_dict(tuple1, tuple2, obj):
    # Create a dictionary such that elements of tuple1 serve as the keys for elements of tuple2.
    dic = dict(zip(tuple1, tuple2))
    # Print the value of the dictionary mapped by obj.
    # print(dic.get(obj))
    print("the value of the dictionary mapped by ", obj, "is: ", dic[obj])
    # Delete the dictionary pairing with the obj key.
    del dic[obj]
    # Print the length of the dictionary.
    print("the length of the dictionary is: ", len(dic))
    # Add a new pairing to the dictionary mapping from obj to the value 0.
    dic[obj] = 0
    # Return a list in which each element is a two-tuple of the dictionary's key-value pairings.
    return list(dic.items())  # replace this


if __name__ == "__main__":
    # Test case
    print(manip_list(["artificial", "intelligence", "rocks"], [4701, "is", "fun"]))
    print("\n")

    try:
        manip_tuple("oh", "no")
    except TypeError:
        print("Can't modify a tuple!")
    print("\n")

    manip_set(["sets", "have", "no", "duplicates"], ["sets", "operations", "are", "useful"], "yeah!")
    print("\n")

    print(manip_dict(("list", "tuple", "set"), ("ordered, mutable", "ordered, immutable", "non-ordered, mutable"),
                     "tuple"))
    print("\n")
