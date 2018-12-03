import numpy as np
import copy
import pickle


# Data loading related

def load_from_pickle(filename, n_jets):
    jets = []
    fd = open(filename, "rb")

    for i in range(n_jets):
        jet = pickle.load(fd)
        jets.append(jet)

    fd.close()

    return jets


# Jet related

# Get the 4-vector pT
def _pt(v):
#     pz = v[2]
#     p = (v[0:3] ** 2).sum() ** 0.5
#     eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
#     pt2 = p / np.cosh(eta)
    pt=(v[0:2] ** 2).sum() ** 0.5
#     print('pt2=',pt2)
#     print('pt=',pt)
    
    return pt



def permute_by_pt(jet, root_id=None):
    # ensure that the left sub-jet has always a larger pt than the right

    if root_id is None:
        root_id = jet["root_id"]

    if jet["tree"][root_id][0] != -1: # because if it is -1, then there are no children and so it is a leaf (SM)
        left = jet["tree"][root_id][0] #root_id left child. This gives the id of the left child (SM)
        right = jet["tree"][root_id][1] #root_id right child. This gives the id of the right child (SM)
        #So object["tree"][root_id] contains the position of the left and right children of object in jet["content"] (SM)

        pt_left = _pt(jet["content"][left]) #We get the pt of the left child (SM)
        pt_right = _pt(jet["content"][right]) #We get the pt of the right child (SM)
#         print('pt_left=',pt_left)
#         print('pt_right=',pt_right)

        if pt_left < pt_right: # We switch the left and right children if necessary so that the left child has greater pt (SM) 
            jet["tree"][root_id][0] = right
            jet["tree"][root_id][1] = left
            print('pt_left=',pt_left)
            print('pt_right=',pt_right)

        # We call the function recursively to ensure each subtree satisfies this property (SM)    
        permute_by_pt(jet, left) 
        permute_by_pt(jet, right)

    return jet


def rewrite_content(jet):
    jet = copy.deepcopy(jet)
#     print('/////'*20)
#     print('jet=',jet)
    
    
#     if jet["content"].shape[1] == 5: #jet["content"].shape[0] is the sequence of subjets in the clustering algorithm. jet["content"].shape[1] is (px,py,pz,E,..) (SM)
#         pflow = jet["content"][:, 4].copy() #All rows and 5th column ? (SM)

    content = jet["content"]
    tree = jet["tree"]

    def _rec(i):
        if tree[i, 0] == -1:
            pass
        else:
            _rec(tree[i, 0])
            _rec(tree[i, 1])
            c = content[tree[i, 0]] + content[tree[i, 1]] #We add the 4-vectors of the two children of object i (SM)
            print('/////'*20)
            print('content[i] before=',content[i])
            content[i] = c # We replace the 4-vector of object i by the sum of the 4-vector of its children
            print('content[i] after=',content[i])
    _rec(jet["root_id"])

#     if jet["content"].shape[1] == 5:
#         jet["content"][:, 4] = pflow
#     print('/////'*20)
#     print('/////'*20)
#     print('jet=',jet)
    return jet

# 
# # We redefine the jet content in terms of eta,phi,p, etc (SM)
# def extract(jet, pflow=False):
#     # per node feature extraction
# 
# #     jet = copy.deepcopy(jet)
# 
#     s = jet["content"].shape
# 
#     if not pflow:
#         content = np.zeros((s[0], 7))
#     else:
#         # pflow value will be one-hot encoded
#         content = np.zeros((s[0], 7+4))
# 
#     for i in range(len(jet["content"])):
#         px = jet["content"][i, 0]
#         py = jet["content"][i, 1]
#         pz = jet["content"][i, 2]
# 
#         p = (jet["content"][i, 0:3] ** 2).sum() ** 0.5
#         eta = 0.5 * (np.log(p + pz) - np.log(p - pz)) #pseudorapidity. 
#         theta = 2 * np.arctan(np.exp(-eta)) #angle with respect to the beam axis. np.arctan is the trigonometric inverse tangent, element-wise. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). 
# #         pt = p / np.cosh(eta)
#         phi = np.arctan2(py, px) # np.arctan2 it the element-wise arc tangent of x1/x2 choosing the quadrant correctly (starting from the x axis).
#         pt = (jet["content"][i, 0:2] ** 2).sum() ** 0.5
# 
# 
#         content[i, 0] = p # Absolute value of the momentum
#         content[i, 1] = eta if np.isfinite(eta) else 0.0  #pseudorapidity
#         content[i, 2] = phi # Azimuthal angle
#         content[i, 3] = jet["content"][i, 3] #Energy
#         content[i, 4] = (jet["content"][i, 3] /
#                          jet["content"][jet["root_id"], 3]) #Energy/ jet energy
#         content[i, 5] = pt if np.isfinite(pt) else 0.0 #Transverse momentum
#         content[i, 6] = theta if np.isfinite(theta) else 0.0  #Angle with respect to the beam axis
# 
#         if pflow:
#             if jet["content"][i, 4] >= 0:
#                 content[i, 7+int(jet["content"][i, 4])] = 1.0
# 
#     jet["content"] = content
# #     print('jet["content"][0:2] =',jet["content"][0:2] )
# 
#     return jet



# We redefine the jet content in terms of eta,phi,p, etc (SM)
def extract(jet, features, pflow=False,kappa=None):
    # per node feature extraction

#     jet = copy.deepcopy(jet)

    s = jet["content"].shape
    
    content = np.zeros((s[0], int(features)))

#     if not pflow:
#         content = np.zeros((s[0], 7))
#     else:
#         # pflow value will be one-hot encoded
#         content = np.zeros((s[0], 7+4))

#     jetpx = jet["content"][jet["root_id"], 0]
#     jetpy = jet["content"][jet["root_id"], 1]
#     jetpz = jet["content"][jet["root_id"], 2]
#     jetpT=(jet["content"][jet["root_id"], 0:2] ** 2).sum() ** 0.5
#     jetp = (jet["content"][jet["root_id"], 0:3] ** 2).sum() ** 0.5
#     jetEta=0.5 * (np.log(jetp + jetpz) - np.log(jetp - jetpz))
#     jetPhi=np.arctan2(jetpy, jetpx)

#     print('/////'*20)
#     print('Jet 4vec=',jet["content"][jet["root_id"]])
#     print('Jet pT=',jet["pt"])
#     print('Jet pT check=',jetpT)
#     print('----'*20)


    for i in range(len(jet["content"])):
        px = jet["content"][i, 0]
        py = jet["content"][i, 1]
        pz = jet["content"][i, 2]

        p = (jet["content"][i, 0:3] ** 2).sum() ** 0.5
        eta = 0.5 * (np.log(p + pz) - np.log(p - pz)) #pseudorapidity. 
        theta = 2 * np.arctan(np.exp(-eta)) #angle with respect to the beam axis. np.arctan is the trigonometric inverse tangent, element-wise. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). 
#         pt = p / np.cosh(eta)
        phi = np.arctan2(py, px) # np.arctan2 it the element-wise arc tangent of x1/x2 choosing the quadrant correctly (starting from the x axis).
        pt = (jet["content"][i, 0:2] ** 2).sum() ** 0.5


#         q=(1/pt**kappa) Qi pt**kappa
        


#         content[i, 0] = p/jetpT # Absolute value of the momentum
#         content[i, 1] = eta if np.isfinite(eta) else 0.0  #pseudorapidity
#         content[i, 2] = phi # Azimuthal angle
# #         content[i, 3] = jet["content"][i, 3] #Energy
#         content[i, 3] = (jet["content"][i, 3] /jetpT) #Energy/ jet pT
#         content[i, 4] = pt/jetpT if np.isfinite(pt) else 0.0 #Transverse momentum
#         content[i, 5] = theta if np.isfinite(theta) else 0.0  #Angle with respect to the beam axis

#         content[i, 0] = p # Absolute value of the momentum 
#         content[i, 1] = p/jetpT # Absolute value of the momentum  
#         content[i, 2] = pt if np.isfinite(pt) else 0.0 #Transverse momentum
#         content[i, 3] = pt/jetpT if np.isfinite(pt) else 0.0 #Transverse momentum  
#         content[i, 4] = jet["content"][i, 3] #Energy/ jet pT
#         content[i, 5] = (jet["content"][i, 3] /jetpT) #Energy/ jet pT   
#         content[i, 6] = phi # Azimuthal angle
#         content[i, 7] = eta if np.isfinite(eta) else 0.0  #pseudorapidity
#         content[i, 8] = theta if np.isfinite(theta) else 0.0  #Angle with respect to the beam axis
        
#         content[i, 9] = ((phi-jetPhi)**2+(eta-jetEta)**2)  #C/A distance with respect to the jet axis. We do not divide by R**2 given that it is just a constant
#         content[i, 10] = np.minimum(pt**2,jetpT**2)*((phi-jetPhi)**2+(eta-jetEta)**2)  #kt distance with respect to the jet axis. We do not divide by R**2 given that it is just a constant
#         content[i, 11] = np.minimum(1/(pt**2),1/(jetpT**2))*((phi-jetPhi)**2+(eta-jetEta)**2)  #anti-kt distance with respect to the jet axis. We do not divide by R**2 given that it is just a constant
        
        content[i, 0] = p # Absolute value of the momentum
        content[i, 1] = eta if np.isfinite(eta) else 0.0  #pseudorapidity
        content[i, 2] = phi # Azimuthal angle
        content[i, 3] = jet["content"][i, 3] #Energy
        content[i, 4] = (jet["content"][i, 3] /
                         jet["content"][jet["root_id"], 3]) #Energy/ jet energy
        content[i, 5] = pt if np.isfinite(pt) else 0.0 #Transverse momentum
        content[i, 6] = theta if np.isfinite(theta) else 0.0  #Angle with respect to the beam axis



#         if pflow:
#             if jet["content"][i, 4] >= 0:
#                 content[i, 7+int(jet["content"][i, 4])] = 1.0

    jet["content"] = content
#     print('jet["content"][0:2] =',jet["content"][0:2] )

    return jet

def randomize(jet):
    # build a random tree

    jet = copy.deepcopy(jet)

    #We get the leaves
    leaves = np.where(jet["tree"][:, 0] == -1)[0] # This gives the positions of jet["tree"] that correspond to a leaf (SM)
    nodes = [n for n in leaves]
    content = [jet["content"][n] for n in nodes]
    nodes = [i for i in range(len(nodes))]
    tree = [[-1, -1] for n in nodes]
    pool = [n for n in nodes]
    next_id = len(nodes)

    while len(pool) >= 2:
        #We randomly pick 2 elements of pool and then delete them (SM)
        i = np.random.randint(len(pool))
        left = pool[i]
        del pool[i]
        j = np.random.randint(len(pool))
        right = pool[j]
        del pool[j]

        nodes.append(next_id)
        c = (content[left] + content[right]) #We get the next object by adding the 4-vector of the children

        if len(c) == 5:
            c[-1] = -1

        content.append(c) #We append the new object to the jet contents (SM)
        tree.append([left, right]) # We append the labels of the left and right children of the new object to the tree (SM)
        pool.append(next_id) # We add the location of the new object (in contents) to the pool of particles we pick from to do the clustering (SM)
        next_id += 1

    # We create the jet (SM)
    jet["content"] = np.array(content)
    jet["tree"] = np.array(tree).astype(int)
    jet["root_id"] = len(jet["tree"]) - 1 # The last element in this case will be the root of the jet tree (SM)

    return jet

# We do the same as in the previous function but sorting by pT in this case (SM) 
def sequentialize_by_pt(jet, reverse=False):
    # transform the tree into a sequence ordered by pt

    jet = copy.deepcopy(jet)

#     print('jet["tree"]=',jet["tree"])
#     print('---'*20)

    leaves = np.where(jet["tree"][:, 0] == -1)[0] #The entries give the position of the leaves in the content list 
#     print('leaves=',leaves)
#     print('---'*20)
    
    nodes = [n for n in leaves]# Same as leaves
#     print('nodes=',nodes)
#     print('---'*20)    
    
    content = [jet["content"][n] for n in nodes] #Here we get the leaves content
#     print('content=',content)
#     print('---'*20)  
        
    nodes = [i for i in range(len(nodes))]
    tree = [[-1, -1] for n in nodes] #We make a tree of length=Number of leaves, where the entries are all (-1), as expected for leaves
#     print('tree=',tree)
#     print('---'*20)     
    
    #Order the list of leaves based on pT. reverse=False gives the list in ascending pT
    pool = sorted([n for n in nodes],
                  key=lambda n: _pt(content[n]),
                  reverse=reverse)
#     print('pool=',pool)
#     print('Ordered pT content=',[ _pt(content[n]) for n in pool])
#     print('---'*20)                  
                  
    next_id = len(pool)

    #Remake the tree. We start from the end of the ordered list of leaves. So in reverse=False we cluster first the greatest pT constituents. => reverse=False gives a tree ordered in descending order in pT
    while len(pool) >= 2:
        right = pool[-1]
        left = pool[-2]
#         print('right=',_pt(content[right]))
#         print('left=',_pt(content[left]))
#         print('---'*20)       
        del pool[-1]
        del pool[-1]

        # We build the tree as a ladder. So we append the right subjet to the list 
        nodes.append(next_id)
#         print('next_id=',next_id)
#         print('---'*20)        
        c = (content[left] + content[right])

        if len(c) == 5:
            c[-1] = -1

        content.append(c) # We append the new subjet to the contents list
#         print('pT c=',_pt(c))
#         print('pT content[next_id]=c?',_pt(content[next_id]))
        tree.append([left, right]) # We append the children locations of the new subjet (locations in the contents list)
#         print('tree=',tree)
#         print('tree[next_id]=',tree[next_id])
#         print('---'*20)
#         print('---'*20)
        pool.append(next_id) #We append the location of the new subjet 'c', so content[next_id]=c
        next_id += 1

    jet["content"] = np.array(content)
    jet["tree"] = np.array(tree).astype(int)
    jet["root_id"] = len(jet["tree"]) - 1

    return jet
