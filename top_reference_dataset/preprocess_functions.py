#!/usr/bin/env python

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import copy
import ROOT as r

# PYTHONPATH
sys.path.append("/het/p4/macaluso/fastjet-install/lib/python2.7/site-packages")
import analysis_functions as af
import fastjet as fj
import tree_cluster_hist as cluster_h
#-------------------------------------------------------------------------------------------------------------
#/////////////////////    AUXILIARY FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
def make_pseudojet(event):
  event_particles=[]
  for n,t in enumerate(event): 
  
#         print('t=',t)
        # Data Format:(E,px,py,pz)
        # FastJet pseudojet format should be: (px,py,pz,E)
        temp=fj.PseudoJet(t[1],t[2],t[3],t[0])

        
#         temp=fj.PseudoJet(t[2],t[3],t[1],t[0])
#         temp=fj.PseudoJet(t[1],t[3],t[2],t[0])
        
        event_particles.append(temp)
        
        
        
#         print('temp.pepr()=',temp.perp())
#         print('temp.mass()=',temp.m())
#         print('temp2.pepr()=',temp2.perp())
#         print('temp2.mass()=',temp2.m())
#         print('==++'*20)
  return event_particles
  

def recluster(particles, Rjet,jetdef_tree):
  
  #----------------------------------------------
  # Recluster the jet constituents and access the clustering history
  #set up our jet definition and a jet selector
  if jetdef_tree=='antikt':
    tree_jet_def = fj.JetDefinition(fj.antikt_algorithm, Rjet)
  elif jetdef_tree=='kt':
    tree_jet_def = fj.JetDefinition(fj.kt_algorithm, Rjet)
  elif jetdef_tree=='CA':
    tree_jet_def = fj.JetDefinition(fj.cambridge_algorithm, Rjet)
  else:
    print('Missing jet definition')
        
#   selector = fj.SelectorPtMin(20.0) #We add extra constraints
#   out_jet = selector(fj.sorted_by_pt(tree_jet_def(preprocess_const_list)))[0] #Apply jet pT cut of 20 GeV and sort jets by pT. Recluster the jet constituents, they should give us only 1 jet if using the original jet radius.
   
  out_jet = fj.sorted_by_pt(tree_jet_def(particles)) #Recluster the jet constituents, they should give us only 1 jet if using the original jet radius
#       print( 'jets=',jets)
#       print('jets const=',jets[0].constituents())
#       print('----'*20) 
      
#       print('Out_jet=',out_jet.px(),out_jet.py(),out_jet.pz(),out_jet.e(),out_jet.m(),out_jet.perp())

  return out_jet



def make_tree_list(out_jet):
      # Create the lists with the trees
  jets_tree=[]
  for i in range(len(out_jet)):
  
    
  
    tree, content, charge,abs_charge,muon= cluster_h._traverse(out_jet[i], extra_info=False)
    tree=np.asarray([tree])
    tree=np.asarray([np.asarray(e).reshape(-1,2) for e in tree])
    content=np.asarray([content])
    content=np.asarray([np.asarray(e).reshape(-1,4) for e in content])
#     print('Content =',content)
    
    mass=out_jet[i].m()
    pt=out_jet[i].pt() 
    
    jets_tree.append((tree, content, mass, pt))
    
    if i>0: print('More than 1 reclustered jet') 
    
    return jets_tree


def make_dictionary(tree,content,mass,pt,charge=None,abs_charge=None,muon=None):
  
  jet = {}

  jet["root_id"] = 0
  jet["tree"] = tree[0] #Labels for the jet constituents in the tree 
  #             jet["content"] = np.reshape(content[i],(-1,4,1)) #Where content[i][0] is the jet 4-momentum, and the other entries are the jets constituents 4 momentum. Use this format if using TensorFlow
  jet["content"] = np.reshape(content[0],(-1,4)) # Use this format if using Pytorch
  jet["mass"] = mass
  jet["pt"] = pt
  jet["energy"] = content[0][0, 3]


  px = content[0][0, 0] #The jet is the first entry of content. And then we have (px,py,pz,E)
  py = content[0][0, 1]
  pz = content[0][0, 2]
  p = (content[0][0, 0:3] ** 2).sum() ** 0.5
  #         jet["Calc energy"]=(p**2+mass[i]**2)**0.5
  eta = 0.5 * (np.log(p + pz) - np.log(p - pz)) #pseudorapidity eta
  phi = np.arctan2(py, px)

  jet["eta"] = eta
  jet["phi"] = phi

  if charge:
    jet["charge"]=charge[0]
    jet["abs_charge"]=abs_charge[0]
  if muon:
    jet["muon"]=muon[0]
    
  return jet
#-------------------------------------------------------------------------------------------------------------
#/////////////////////    NYU PREPROCESSING     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
 
# THIS IS MY OWN IMPLEMENTATION OF THE PREPROCESSING IN LOUPPE ET AL 2017. (
#I found 2 bugs in that implementation:
# 1) They should find alpha after rotate in the z axis with phi and boost
# 2) They should find sj3.pz() after rotate in the z axis with phi, boost and rotate in the x axis with alpha

# Preprocessing algorithm:
# 1. j = the highest pt anti-kt jet (R=1)
# 2. run kt (R=0.3) on the constituents c of j, resulting in subjets sj1, sj2, ..., sjN (We get the subjets)
# 3. phi = sj1.phi(); for all c, do c.rotate_z(-phi) (We center the leafing subjet at phi=0)
# 4. bv = sj1.boost_vector(); bv.set_perp(0); for all c, do c.boost(-bv) (We boost so that the leading subjet is at eta=0)
# 5. deltaz = sj1.pz - sj2.pz; deltay = sj1.py - sj2.py; alpha = -atan2(deltaz, deltay); for all c, do c.rotate_x(alpha) (We rotate so that the 2nd leading subjet is at 12 o'clock)
# 6. if sj3.pz < 0: for all c, do c.set_pz(-c.pz) (We flip so that the 3rd leading subjet is on the right half plane)
# 7. finally recluster all transformed constituents c into a single jet (using kt or anti-kt? r?) (We recluster the jets after transforming)
def get_alpha(subjets,phi,boost):

    temp_subjets=[af.tempvector(subjet.perp(),subjet.eta(),subjet.phi_std(),subjet.m()) for subjet in subjets]
#     print('Subjets [mass,pT,eta,phi] before =',[[temp_subjet.M(),temp_subjet.Perp(),temp_subjet.Eta(), temp_subjet.Phi()] for temp_subjet in temp_subjets])

# 
#     # Get alpha (after rotation and boost) 
#     if len(temp_subjets)>=2:    
#       deltaz_before = temp_subjets[0].Pz() - temp_subjets[1].Pz()
#       deltay_before = temp_subjets[0].Py() - temp_subjets[1].Py()
# 
#       alpha_before = -np.arctan2(deltaz_before, deltay_before)
#       print('alpha_before=',alpha_before)

    for i,temp_subjet in enumerate(temp_subjets[0:3]):
              
      temp_subjet.RotateZ(-phi)
      temp_subjet.Boost(-boost)
#       print('Subjets [mass,pT,eta,phi] after=',[[temp_subjet.M(),temp_subjet.Perp(),temp_subjet.Eta(), temp_subjet.Phi()] for temp_subjet in temp_subjets]) 
#       print('---'*20)
#       print('---'*20) 
      
    #-----------------  
    # Get alpha (after rotation and boost) 
    if len(temp_subjets)>=2:    
      deltaz = temp_subjets[0].Pz() - temp_subjets[1].Pz()
      deltay = temp_subjets[0].Py() - temp_subjets[1].Py()

      alpha = -np.arctan2(deltaz, deltay)
#       print('alpha after=',alpha)

  #     print('Subjets [mass,pT,eta,phi] after=',[[temp_subjet.M(),temp_subjet.Perp(),temp_subjet.Eta(), temp_subjet.Phi()] for temp_subjet in temp_subjets]) 
  #     print('---'*20)
  #     print('---'*20) 
    
      #---------------- 
      #Rotate subjet 3
      if len(temp_subjets)>=3:
        temp_subjets[2].RotateX(alpha)
      
        return alpha, temp_subjets[2].Pz()

      else:
      
        return alpha






#------------------------------------------------
def preprocess_nyu(subjets):

    ##Comment: Subjets are FastJet Pseudojets and constituents are TLorentz vectors
#     print(' Number of subjets=',len(subjets))
#     print('Subjets [mass,pT,eta,phi,pz]=',[[subjet.m(),subjet.perp(),subjet.eta(), subjet.phi_std(),subjet.pz()] for subjet in subjets])
    
    #------------------------
    # 1) phi = sj1.phi(); for all c, do c.rotate_z(-phi) (We center in phi so that the leading subjet is at phi=0)
    # for all c, do c.rotate_z(-phi) 
    phi=subjets[0].phi_std()
#     print('Phi=',phi)
    
    #------------------------
    # 2) Find boost so that the leading subjet in pT is at eta=0 (pz=0)
    lead_subjet=af.tempvector(subjets[0].perp(),subjets[0].eta(),subjets[0].phi_std(),subjets[0].m())
    boost=lead_subjet.BoostVector()
    boosted= r.TLorentzVector()
#     print('boost [pT, eta, phi] before =',[boost.Perp(),boost.Eta(),boost.Phi()])
#     print('boost [px, py, pz] before =',[boost.Px(),boost.Py(),boost.Pz()])
    boost.SetPerp(0.0) # We set the boost pT to 0 so that we only boost on the z axis (we have already rotated in phi so we don't want to change (px,py) of the constituents
#     print('boost [pT, eta, phi] after =',[boost.Perp(),boost.Eta(),boost.Phi()])
#     print('boost [px, py, pz] after =',[boost.Px(),boost.Py(),boost.Pz()])
#     print('---'*20)
    
      
    #------------------------
    # 3) We rotate so that the 2nd leading subjet is at 12 o'clock, eta=0 (pz=0)
    # deltaz = sj1.pz - sj2.pz
    # deltay = sj1.py - sj2.py
    # alpha = -atan2(deltaz, deltay)
    # for all c, do c.rotate_x(alpha)
    
    # Then we flip the constituents along the z axis if the 3rd leading subjets (after the transformations) is on the left hand side,i.e. subj3Pz<0
    
    if len(subjets) >= 3:
      alpha, subj3Pz= get_alpha(subjets,phi,boost)
    
    elif len(subjets) == 2:
      alpha = get_alpha(subjets,phi,boost)
    
    # After getting phi, boost and alpha we apply the transformations on the jet constituetns 
    out_const=[] 
    for subjet in subjets:
      for const in subjet.constituents():
      
        #Make the constituents TLorentz vectors
        temp=af.tempvector(const.perp(),const.eta(),const.phi_std(),const.m())
#         print('Temp after phi rot [pT,eta,phi,m,pz,E] before=',[temp.Perp(),temp.Eta(),temp.Phi(),temp.M(),temp.Pz(),temp.E()])
        
        #Rotate
        temp.RotateZ(-phi)
#         print('Temp [pT,eta,phi,m,pz] before=',[temp.Perp(),temp.Eta(),temp.Phi(),temp.M(),temp.Pz()])
      
        #Boost
        temp.Boost(-boost) 
#         print('Temp after boost [pT,eta,phi,m,pz,E] before=',[temp.Perp(),temp.Eta(),temp.Phi(),temp.M(),temp.Pz(),temp.E()])

        
        # Rotate so that 2nd max is at 12 o'clock (Lorentz transformations contain the rotations in R^3)
        if len(subjets) >= 2:      
          temp.RotateX(alpha)
#           print('Temp after X rot [pT,eta,phi,m,pz,E] before=',[temp.Perp(),temp.Eta(),temp.Phi(),temp.M(),temp.Pz(),temp.E()])
      
        #  Flip so that the 3rd leading subjet is on the right half plane.(Switch eta -> -eta)
        if len(subjets) >= 3 and subj3Pz < 0:
          temp.SetPz(-temp.Pz())
#           print('Temp [pT,eta,phi,m,pz] before=',[temp.Perp(),temp.Eta(),temp.Phi(),temp.M(),temp.Pz()])
          
#         print('+-+-'*20)
        # Output preprocessed constituents
        out_const.append(fj.PseudoJet(temp.Px(),temp.Py(),temp.Pz(),temp.E()))

    return out_const



















#------------------------------------------------------------------------------------------------------------- 
# Function to plot the histograms
def makeHist(out_dir,data,bins,plotname,title,xaxis,yaxis,type,Njets):
  myfig = plt.figure()
  ax1 = myfig.add_subplot(1, 1, 1)
  n, bins, patches = ax1.hist(data,bins,alpha=0.5)
  ax1.set_xlabel(str(xaxis))
  ax1.set_ylabel(str(yaxis))
  ax1.set_title('Histogram of '+str(title))
  ax1.grid(True)
  plot_FNAME = 'Hist_'+str(plotname)+'_'+type+'_'+str(Njets)+'.png'
  print('------------'*10)
  print('Hist plot = ',out_dir+'/'+plot_FNAME)
  print('------------'*10)
  plt.savefig(out_dir+'/'+plot_FNAME)

def make2DHist(out_dir,data1,data2,bins,plotname,title,xaxis,yaxis,type,Njets,xmin=None,xmax=None,ymin=None, ymax=None):
  myfig = plt.figure()
  ax1 = myfig.add_subplot(1, 1, 1)
#   fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True,tight_layout=True)
#   n, bins, patches = ax1.hist2d(data1,data2,bins)
  ax1.hist2d(data1,data2,bins,range=[[xmin, xmax], [ymin, ymax]])
  ax1.set_xlabel(str(xaxis))
  ax1.set_ylabel(str(yaxis))
  ax1.set_title('Histogram of '+str(title))
  ax1.grid(True)
  plot_FNAME = 'Hist_'+str(plotname)+'_'+type+'_'+str(Njets)+'.png'
  print('------------'*10)
  print('Hist plot = ',out_dir+'/'+plot_FNAME)
  print('------------'*10)
  plt.savefig(out_dir+'/'+plot_FNAME)

#-------------------------------------------------------------------------------------------------------------- 












