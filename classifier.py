# classifier.py
#
# Sean, Kyle, Tristan
# Prof. Fitzsimmons
# Artificial Intelligence
# Project 3 
# 
# Description: Decision-Tree Learning that prints decision trees given a set of files
# 
# Usage: python3 classifier.py <attributes> <<training-set> <testing-set> <significance>
# 

import csv
import sys
import math
import scipy.stats

#  Nodes for our tree building 
class Node:
  def __init__(self, value= None, parent=None, children= {}, attributes= []):
    self.value= value
    self.parent= parent
    self.children= children
    self.attributes= attributes

def read_file(filename):  # read file
    print("reading " + filename)
    test = []
    if filename[:-3] == "csv": # if the file is a csv , then go here
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='|', quotechar=' ')
            for row in spamreader:
                test.append(row)
                print(row)
        return test
    else: # otherwise if txt
        attribute_list = [] 
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                templist = line.split(",")
                attribute_list.append(templist)
            
        return attribute_list

def importance(attributes,rowList,labels): #importance function that calculates
    max= 0                 # initialize maximum entropy to 0
    maxNode= 0             # initialize maximum entropy column index to 0
    cols= []               # initialize list to store columns of data
    response_col= []       # initialize list to store response column of data

    for row in rowList: # iterate over rows in data and add first element of each row to response_col
        response_col.append(row[0])

   
    for i in labels[1:]: # iterate over columns in data, excluding the response column
        index = labels.index(i) # get index of column in labels
        newCol = []             # initialize list to store column data
        
        for row in rowList:  # iterate over rows in data and add the element at the column index to newCol
            newCol.append(row[index])
        cols.append(newCol) # add newCol to cols list

    #print(labels) # print column labels (for debugging)

   
    for i in range(1,len(labels)): # iterate over columns in cols
        #print("Label: "+labels[i])  # print label and column data for current column (for debugging)
        #print("i: "+str(i) + " cols "+str(cols))

        currEntropy= entropy(response_col,cols[i-1], labels)# calculate entropy for current column using entropy function

        if max < currEntropy: # if current entropy is greater than maximum entropy, update max and maxNode
            max= currEntropy
            maxNode= i

    return max, maxNode # return maximum entropy and column index with maximum entropy



def entropy(responses, column, labels): #Remainder function (just named entropy for fun)

    responses_dict = {i:responses.count(i) for i in responses if i not in labels} 
    p_n_dict = {i:column.count(i) for i in column if i not in labels} # get a dictionary of amount of yes and no in list  
  
    pk= [0]* len(p_n_dict)
    pos= responses[0] # positive value will be the first entry in the responses
   
    ex= []                 # initialize list to store column values
   
    for i in p_n_dict.keys():  # iterate over keys in p_n_dict and add each key to ex
        ex.append(i) 
    # print("ex", ex) # print ex for debugging
    response_ex= []       # initialize list to store response values
    for i in responses_dict.keys(): # iterate over keys in responses_dict and add each key to response_ex
        response_ex.append(i)

    for i in range(len(responses)):
        if (responses[i]== pos): # if current response is equal to pos, increment the value at the index of the corresponding column value in pk
            pk[ex.index(column[i])]+=1

    num= responses_dict.get(response_ex[0]) # get number of occurrences of first response value in responses_dict

    denom= 0
  
    for i in response_ex:  # iterate over elements in response_ex and add the number of occurrences of each response value to denom
        denom+= int(responses_dict.get(i))


    remainder= 0
    for i in ex:
        j= ex.index(i) # get index of current element in ex

        remainder += ((p_n_dict.get(i))/denom) *b(pk[j]/p_n_dict.get(i))       # add the product of the relative frequency of the current element in p_n_dict and the binary entropy of the relative frequency of the current element in pk to remainders


    importanceVal= b(float(num)/float(denom))- remainder # follows the formula from class
    #print("importanceVal ", importanceVal) #print for debugging

    return importanceVal

def b(q): #entropy function 
    if q== 0 or q== 1:
        return 0
    else:
        return (-(q* math.log2(q) + (1-q)*math.log2(1-q)))


def plurality(examples): # plurality function
    value_counts = {} # Count the number of occurrences of each value in the given column
    for row in examples:
        value = row[0]
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1 
    return max(value_counts, key=value_counts.get)

def printTree(rootNode, depth,total_nodes, decision_nodes,test): # print the decision tree 
        
        if rootNode.value not in test: #if node value is unique add it to a growing list
            test.append(rootNode.value)
        else: #if node has already been used, then return
            return total_nodes, decision_nodes, depth
        
        print('\t'*depth, end = " ") # format prints 
        print("Testing", rootNode.value)
       
        for branch in rootNode.attributes: # for each branch in the the node (branch is the key to the dict)
            print('\t'*depth, end = " ")   # look at the children for each branch and recurse if it is not a leaf node
            print("Branch " + branch)
            if rootNode.children == {}: # check if has children
                return total_nodes, decision_nodes, depth
            
            elif isLeaf(rootNode.children[branch]): #check for leaf node
                print('\t'*depth, end= " ")
                print("Leaf with value:",rootNode.children[branch].value)
            else: #else recurse 
                total, decision, maXdepth = printTree(rootNode.children[branch].value,depth+1, total_nodes+1,decision_nodes+1,test) # recurse 

        return total, decision, maXdepth

#checks if a node is a leaf
def isLeaf(node): 
    return isinstance(node.value, str)

def chi2_prune(tree, alpha):

    for child in tree.children:
        if not isLeaf(child):
            chi2_prune(child, alpha)
        else:
            break
    pkhat=0
    nkhat=0
    pk=[]
    nk=[]
    p=0
    n=0

    for child in tree.children:
        for i in child.attributes: 
            pk.append([0])
            nk.append([0]) # build pk and nk off of all the attributes
            if isLeaf(child[i]):
                if child[i]== 'Yes' or child[i]== "Republican":
                    p+=1
                    pk[i]+=1
                else:
                    n+=1
                    nk[i]+=1
    pkhat= p* ((sum(pk,0) + sum(nk,0))/(p+n)) 
    nkhat= n* ((sum(pk,0) + sum(nk,0))/(p+n)) # calculate pkhat and nkhat from the formula in slides

    delta= []
    for i in range(len(pk)):
        delta.append((((pk[i]-pkhat)**2)/pkhat) + (((nk[i]-nkhat)**2)/nkhat)) # calculate delta from formula in slides
    deltaval= sum(delta,0) # sum delta values from 0-length

    df= len(pk)-1
    chi2val = scipy.stats.chi2.ppf(1-alpha, df) # function from hw
    if(deltaval<chi2val): # prune node and set it to a leaf node
        tree.children={}
        tree.value= plurality(tree.parent)
        tree.attributes= [] 
    else: # stop pruning and return tree
        return tree 

    chi2_prune(tree.parent,alpha) # call chi2 on parent



# Build a decision tree given a list of attribute, examples, labels, and parent_examples
def decision_tree_builder(attributes, examples, labels, parent_examples): # decision tree builder 
    #attributeCopy = attributes.copy() # get a copy of the  list
    classification = []
    for row in examples: # get the classification stats beforehand
        classification.append(row[0])
    
    # Base Cases 
    if examples == []: # empty list
        return plurality(parent_examples)
    elif (len(set(classification))==1) : #have the same classification
        return classification[0] 
    elif attributes == []: #attribute list empty
        return plurality(examples)

    # Make decision tree recursively 
    else:  
        importanceVal, selectIndex= importance(attributes, examples, labels) # get the importance values
        #print("selected attribute", selectIndex, "which is", labels[selectIndex], "with an importance value of", importanceVal) # print for debugging 
        tree= Node() #intialize head node
        tree.value = labels[selectIndex] 
        tree.attributes= attributes[selectIndex][1:]
        for i in attributes[selectIndex][1:]: # go through the data and add based on importance 
            copy_examples=[]
            for rows in examples:
                if rows[selectIndex]== i:
                    copy_examples.append(rows)
            tree.children[i] = Node(parent = tree, value= decision_tree_builder(attributes, copy_examples, labels, examples))
        return tree 

# chi2 pruning function 
def chi2(tree, significance): 
    print("meow")


def main():
    if len(sys.argv) != 5: # make sure user inputs correct amount of arguments, otherwise exit
        print("Usage: python3 classifier.py <attributes>  <training-set> <testing-set> <significance>")
        sys.exit(1)
    attribute_file = sys.argv[1]
    training_file = sys.argv[2]
    test_file = sys.argv[3]
    significance = sys.argv[4]
    
    # Initialize our data and manipulate for new data 
    attributes = read_file(attribute_file)
    train = read_file(training_file) 
    test = read_file(test_file)
    labels= train[0]
    rows= train[1:]

    #Store tree within one node, then print
    headNode= decision_tree_builder(attributes, rows, labels, rows)
    print("-- Printing Decision Tree --")
    min = 1
    total, decision, max = printTree(headNode, 0, 0,0,[]) 
    # print stats 
    print("Total Nodes: " +str(total*3))
    print("Decision Nodes: " +str(decision*2))
    print("Max Depth: " +str(max))
    print("Min Depth: " + str(min))
    print("Average Depth of Root-to-Leaf: " +str(float((total*3+ decision*2 + min )/(decision*2))))

    #newHeadNode= chi2_prune(headNode, significance)
    #print("-- Printing Pruned Tree --")
    #printTree(newHeadNode, 0, 0, 0, [])
    #print("Total Nodes: " +str(total*3))
    #print("Decision Nodes: " +str(decision*2))
    #print("Max Depth: " +str(max))
    #print("Min Depth: " + str(min))
    #print("Average Depth of Root-to-Leaf: " +str(float((total*3+ decision*2 + min )/(decision*2))))
if __name__ == "__main__":
    main()



