#!/usr/bin/env python

import sys, copy
#pyroot module
# import ROOT as r
# import itertools
# numpy, scipy module
import numpy as np  




# PYTHONPATH
# sys.path.append("/het/p4/macaluso/fastjet-install/lib/python2.7/site-packages")
sys.path.append("/opt/fastjet-install/lib/python2.7/site-packages")

import fastjet as fj


# ----------------------------------------------


#Find DeltaR
def mindeltaRjet_parton(jet,partons):
  DeltaRjmin=100000
  jID=0
  for id,parton in enumerate(partons):
    DeltaRj= abs(jet.DeltaR(parton))
    if DeltaRj<DeltaRjmin:
      DeltaRjmin=DeltaRj
      jID=id
  return DeltaRjmin, jID





#boost particle a into particle b rest frame 
def boost_a_to_b_restframe(a,b):
    boost=b.BoostVector()
    boosted= r.TLorentzVector()
    boosted=a.Boost(-boost)
    return boosted

def get_theta(px,py,pz):
  temp = r.TVector3()
  temp.SetXYZ(px,py,pz)
  return temp.Theta()
# ------------------------------------------------






#------------------------------------------------------------------------
# saves.dat file with (n-dimensional) array (one per line if n>1)
def savedat(var,filename):
	thefile = open(filename,'w')
	if not isinstance(var[0],list):
	    for nn in var:
	        thefile.write(str(nn))
	        thefile.write(' ')
	else:
	    for vv in var:
	        for nn in vv:
	            thefile.write(str(nn))
	            thefile.write(' ')
	        thefile.write('\n')
	thefile.close()


#function which takes in pt/eta/phi/m and returns a
#TLorentzVector
def tempvector(pt,eta,phi,m):
    temp = r.TLorentzVector()
    temp.SetPtEtaPhiM(pt,eta,phi,m)

    return temp

def mtb(bjets,MET):
    phi = []
    for b in bjets:
        phi.append(abs(b.DeltaPhi(MET)))
    ind=np.argmin(phi)
    mtbmin=np.sqrt(2*bjets[ind].Pt()*MET.Pt()*(1-np.cos(phi[ind])))
    ind=np.argmax(phi)
    mtbmax=np.sqrt(2*bjets[ind].Pt()*MET.Pt()*(1-np.cos(phi[ind])))

    return [mtbmin,mtbmax]

