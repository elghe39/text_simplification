from nltk.parse import CoreNLPParser
from nltk.tree import Tree, ParentedTree
import nltk.draw.tree
import copy

parser = CoreNLPParser(url='http://localhost:9000')


def action(sentence: str):
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    tree = next(parser.parse(parser.tokenize(sentence)))
    tree = ParentedTree.convert(tree)
    # nltk.draw.tree.draw_trees(tree)

    treeList = removeComplex(tree)
    # for each in treeList:
    #     print(each)
    res = set()
    temp = []
    for index, node in enumerate(treeList):
        lis = removeConjunction(node)
        for each in lis:
            temp.append(each)
    for i in temp:
        string = " ".join(i.leaves())
        if string[-1] == ',':
            string = string[:-2]
        if string[-1] != '.':
            string += '.'
        if string[0].islower():
            string = string[0].upper() + string[1:]
        res.add(string)
    return res


def removeComplex(tree: Tree):
    subTrees = list()
    for subTree in reversed(list(tree.subtrees())):
        hasNP = False
        if subTree.label() == 'SBAR':
            i = int()
            for index, children in enumerate(subTree):
                i = index
                if children.label() == 'S':
                    for grandChildren in children:
                        if grandChildren.label() == 'NP':
                            # subtree contains the 'NP'
                            hasNP = True
            # don't have to take subject from other part of the sentence
            if hasNP:
                subTrees.append(subTree[i])
                del tree[subTree.treeposition()]
            else:
                if subTree.leaves()[0] in ['that', 'which']:
                    prevs = list()
                    for prev in list(tree.subtrees()):
                        if prev == subTree:
                            break
                        else:
                            if prev.label() == 'NP':
                                prevs.append(prev)
                    np = copy.deepcopy(prevs[-1])
                    newNode = copy.deepcopy(subTree)
                    newNode.insert(0, np)
                    subTrees.append(newNode)
                    del tree[subTree.treeposition()]
                else:
                    tree1 = Tree
                    tree2 = Tree
                    has_tree1 = False
                    has_tree2 = False
                    for index, node in enumerate(subTree.parent()):
                        if node.label() == 'NP':
                            tree1 = node
                            has_tree1 = True
                        elif node.label() == 'VP':
                            for sunIndex, subNode in enumerate(node):
                                if subNode.label()[:2] == 'VB':
                                    tree2 = subNode
                                    has_tree2 = True
                    del tree[subTree[0].treeposition()]
                    if has_tree2:
                        treeTemp2 = copy.deepcopy(tree2)
                        subTree.insert(0, treeTemp2)
                    if has_tree1:
                        treeTemp1 = copy.deepcopy(tree1)
                        subTree.insert(0, treeTemp1)
                    subTrees.append(subTree)
                    del tree[subTree.treeposition()]
    subTrees.append(tree)
    return subTrees


def removeConjunction(tree: Tree):
    lis = helper(tree)
    return lis


def helper(tree: Tree) -> list:
    changed = False
    subTreeList = list()
    result = list()
    for subTree in list(list(tree.subtrees())):
        if subTree.label() == 'CC':
            parent = subTree.parent()
            # print(parent)
            for children in parent:
                if children.label() != 'CC':
                    newNode = copy.deepcopy(tree)
                    for subTree2 in list(list(newNode.subtrees())):
                        if subTree2 == subTree:
                            parent2 = subTree2.parent()
                            # print(parent2)

                    toDelete = True
                    while toDelete:
                        for otherChildren in parent2:
                            if children != otherChildren:
                                del newNode[otherChildren.treeposition()]
                                break
                        if len(parent2) == 1:
                            toDelete = False
                    subTreeList.append(newNode)
            changed = True
            break
    if changed:
        for each in subTreeList:
            newList = helper(each)
            if len(newList) != 0:
                for node in newList:
                    result.append(node)
        return result
    else:
        return [tree]


def main(sentence: str):
    sentence = action(
        "This month was originally named Sextilis in Latin, because it was the sixth month in the ancient Roman calendar, which started in March about 735 BC under Romulus.")
    print(sentence)


if __name__ == '__main__':
    main(".")
