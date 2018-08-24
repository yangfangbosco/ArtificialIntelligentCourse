# CSCI561 Homework1
# Yang Fang
# Jun 12, 2018

import sys

# create the Node Class
class Node(object):
    def __init__(self):
        self.parent = None; # Node
        self.child = None;  # Node[]
        self.owner = None;  #R1 or R2
        self.depth = None;
        self.name = None;
        self.score = None;
        self.child = [];
        self.discovered = 0;
        
        

# Variables
# Parse the input file
inputFile=open(sys.argv[2])
#inputFile=open("myinput7.txt","r")
day = inputFile.readline().rstrip();
#set day
#print day;
# set player
nextPlayer = inputFile.readline().rstrip();
if(nextPlayer == "R1"):
    lastPlayer = "R2";
else:
    lastPlayer = "R1";
#print "next player is " + nextPlayer;
#print "last player is " + lastPlayer;
# RPL
line = inputFile.readline().rstrip();
#print line
RPL_Array =  line.replace(')','').replace('(','').split(','); 
i = 0;
RPL = {};
while i < len(RPL_Array):       
    RPL[RPL_Array[i]] = float(RPL_Array[i+1]);
    i = i + 2;
#if it is yesterday
if(day == "Yesterday"):
    num_region = len(RPL.keys());
    summ = 0.0;
    for v in RPL.values():
        summ += v;
    for k in RPL.keys():
        RPL[k] = (summ/num_region+ RPL[k])/2;
RPL["PASS"] = 0;
#print RPL;

keyArr = RPL.keys();
keyArr.sort();

infinite = 0;
for x in RPL.values():
    infinite += x;
posInfinite = infinite * 10;
negInfinite = -1 * posInfinite;
#print "posinfinite is " + str(posInfinite);
#print "posinfinite is " + str(negInfinite);

# Adjacent Matrix
lineArray = inputFile.readline().replace('[','').replace(']','').rstrip().split(',');
m_neighbor = {};
i = 0;
while(i < len(lineArray)):
    arr = [];
    j = 0;
    while(j<len(lineArray)):
        if(lineArray[j] != "0" and i!=j):
            arr.append(keyArr[j]);
        j = j+1;
    m_neighbor[keyArr[i]] = arr;
    lineArray = inputFile.readline().replace('[','').replace(']','').rstrip().split(',');
    i = i+1;
m_neighbor["PASS"] = [];
#print m_neighbor;


# Picked Region
picked = lineArray;
if(picked[0] == '*'):
    picked = [];
#print "picked" + str(picked);

# Max Depth
maxDepth = int(inputFile.readline().rstrip());
#print "maxDepth is " + str(maxDepth);


# Start constructing the activity search tree       
# determine who start the game
firstPlayer = "";
secondPlayer = "";
if(len(picked) % 2 == 0):
    firstPlayer = nextPlayer;
    secondPlayer = lastPlayer;
else:
    firstPlayer = lastPlayer;
    secondPlayer = nextPlayer;
#construct the root tree
nodeArray = [];
root = Node();
root.name = "root";
root.depth = -1;
root.score = 0;
nodeArray.insert(len(nodeArray), root);

# construct the rest of the picked node
i = 0
while(i < len(picked)):
    nextNode = Node();
    nextNode.parent = nodeArray[-1];
    nextNode.depth = i;
    nextNode.name = picked[i];
    if(i%2 == 0):
        nextNode.owner = firstPlayer;
    else:
        nextNode.owner = secondPlayer;
    if(nextNode.owner == nextPlayer):
        nextNode.score = nextNode.parent.score + RPL[nextNode.name];
    else:     
        nextNode.score = nextNode.parent.score;
    nodeArray.insert(len(nodeArray), nextNode);
    nodeArray[i].child.append(nextNode);
    i = i + 1;
# insert the picked node into frontier and explored set, using our setting, no matter what, the frontier will not be empty
frontier = nodeArray[-1];
explored = nodeArray[0:len(nodeArray)-1];
startingNode = nodeArray[-1];

#printing explored
i = 0
#print "explored:"
while(i < len(explored)):
    #print explored[i].name;
    i = i+1;
    
#printing frontier
i = 0
#print "frontier:"
#print frontier.name;



def getChoices(node):
    #print "processing "+ node.name;
    #print "owner "+ node.owner;
    #determine the child
    node.discovered = 1;
    parentNode = node.parent;
    isPlayer = 1;
    playerOwned = [];
    opponentOwned = [];
    while(parentNode!= None and parentNode.name != "root"):
        if(isPlayer == 1):
            playerOwned.append(parentNode.name);
            isPlayer = 0;
        else:
            opponentOwned.append(parentNode.name);
            isPlayer = 1;
        parentNode = parentNode.parent; 
    if(node.name != root):
        opponentOwned.append(node.name);
    set_avil = set([]);
    if(len(playerOwned)!=0):
        for x in playerOwned:
            for y in m_neighbor[x]:
                    if(y not in set_avil):
                        set_avil.add(y);
    else:
        for x in RPL.keys():
            set_avil.add(x);
        set_avil.remove("PASS")
    
    for x in opponentOwned+playerOwned:
        if(x in set_avil):
            set_avil.remove(x);
            
    arr_avil = list(set_avil);
    arr_avil.sort();
    if(len(arr_avil) == 0 and node.name != "PASS"):
        arr_avil.append("PASS");
   # print playerOwned;
   # print opponentOwned;
   # print arr_avil;
    return arr_avil;

def createNode(parent, name, maxDepth):
    add = Node();
    add.name = name;
    add.depth = parent.depth+1;
    add.parent = parent;
    add.discovered = 0;
    add.child = [];
    if(parent.owner == "R1"):
        add.owner = "R2";
    else:
        add.owner = "R1";
    if(add.owner == nextPlayer):
        add.score = parent.score + RPL[add.name];
    else:
        add.score = parent.score;
    #print "adding " + add.name + " with score of "+ str(add.score) + " depth of "+ str(add.depth)+ " parent is " + add.parent.name +"\n";
    return add;

def DFS(node, maxDepth):
    arr_avil = getChoices(node);
    add_arr=[];
    for x in arr_avil:
        add = createNode(node,x,maxDepth);
        add.child = DFS(add, maxDepth);
        add_arr.append(add);
    return add_arr;
        
     
#print "starting dfs\n"

def mini(a,b):
    if(a.score < b.score):
        return a;
    if(a.score > b.score):
        return b;
    else:
        return a;
    
    
def maxi(a,b):
    if(a.score > b.score):
        return a;
    if(a.score < b.score):
        return b;
    else:
        return a;





def minimaxDFS(node, maxDepth,leaves, alpha, beta):
    arr_avil = getChoices(node);
    if(len(arr_avil) == 0 or node.depth == maxDepth):
        #print node.score;
        #print "this is the leaf " + node.name +"with neighbor length of " + str(len(arr_avil)) +"and depth " + str(node.depth);
        leaves.append(node.score);
        return node;
    if(node.owner == nextPlayer):
        pick = Node();
        pick.name = "posInf";
        pick.score = posInfinite;
        for x in arr_avil:
            add = createNode(node,x,maxDepth);
            result = minimaxDFS(add, maxDepth, leaves, alpha, beta);
            #print "comparing "+ pick.name +" and " + result.name;
            pick = mini(pick, result);
            if(pick.score <= alpha.score):
                break;
            beta = mini(beta, pick);
        #print pick.score;
        return pick;  
    else:
        pick = Node();
        pick.name = "negPos";
        pick.score = negInfinite;
        for x in arr_avil:
            add = createNode(node,x,maxDepth);
            result = minimaxDFS(add, maxDepth, leaves, alpha, beta);
            #print "comparing "+ pick.name +" and " + result.name;
            pick = maxi(pick, result);
            if(pick.score >= beta.score):
                break;
            alpha = maxi(alpha, pick);
            
        #print pick.score;
        return pick;  

    
negInf = Node();
negInf.score = negInfinite;
posInf = Node();
posInf.score = posInfinite;

leaves = [];
#a=alphaBetaDFS(frontier , negInf, posInf,maxDepth, maxDepth);
v = minimaxDFS(frontier, maxDepth,leaves, negInf, posInf)
#DFS(frontier,100);

outputFile = open("output.txt","w");

#print v.name
#print v.score
#print v.parent.name
while(v.parent!=None and v.parent != startingNode):
    v = v.parent;

#print "the final pick is "+ str(v.name);
#print leaves;
outputFile.write(str(v.name) +"\n");  
scorestring = "";
for x in leaves:
    scorestring += str(int(round(x,0))) + ",";
    #print str(int(round(x,0)));
outputFile.write(scorestring[:-1]);      
outputFile.close();
#testing using DFS to return the biggest value 

        

#AlphaBeta(frontier, 1, 2);


