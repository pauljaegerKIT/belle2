#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def getKsStats(df,n):
    """
    Analyze tree: Calulate Efficiencies, Purities, 
    Multiplicities etc. for different selection criteria
    """
    tree="kstree"
    t = df[tree]
    t00 = df["ks00tree"]
    total_prod = int(len(df[getinfo(tree)["truth"]])  * getinfo(tree)["truthfactor"] )
    sig_prod   = int(n * getinfo(tree)["sigfactor"] ) # events to expected signal
    
    tag_prod   = total_prod - sig_prod
    
    total_reco_00 = len(df["ks00tree"][(df["ks00tree"]["K_S0_mcPDG"]==310)& (df["ks00tree"][getinfo(tree)["status"]]==0)])   
    total_reco = len(t[(t["K_S0_mcPDG"]==310)& (t[getinfo(tree)["status"]]==0)])
    sig_reco   = len(t[(t["K_S0_mcPDG"]==310) & ( (t['K_S0_MC_MOTHER_ID']==511) | (t['K_S0_MC_MOTHER_ID']==-511)  ) \
    & (t[getinfo(tree)["status"]]==0)])
    sig_reco_00   = len(t00[(t00["K_S0_mcPDG"]==310) & ( (t00['K_S0_MC_MOTHER_ID']==511) | (t00['K_S0_MC_MOTHER_ID']==-511)  ) \
    & (t00[getinfo(tree)["status"]]==0)])
    
    overall_reco = total_reco + total_reco_00
    overall_sig_reco = sig_reco + sig_reco_00
        
        
    tag_reco   = overall_reco - overall_sig_reco   
    cand_list = len(t)
    cand_list_00 = len(df["ks00tree"])
    
    eff = np.round(sig_reco / float(sig_prod * 0.692),3)
    pur = np.round(sig_reco / float(cand_list),3)
    
    eff_00 = np.round(sig_reco_00 / float(sig_prod * 0.3069),3)
    pur_00 = np.round(sig_reco_00 / float(cand_list_00),3)
    
    overall_eff = np.round(overall_sig_reco / float(sig_prod),3)
    overall_pur = np.round(overall_sig_reco/ float(cand_list+cand_list_00),3)
    
    
    
    multi = df[tree].groupby('evt_no').size().mean() * 1000 / n 
    multi_00 = df["ks00tree"].groupby('evt_no').size().mean() * 1000 / n 
    
    evt = len(df[tree][0:999].groupby('evt_no').size()) * n/ 1000
    evt_00 = len(df["ks00tree"][0:999].groupby('evt_no').size()) * n/ 1000
    
    names=OrderedDict()
    names["total produced"]= total_prod
    names["signal side produced"]= sig_prod
    names["tag side produced"]= tag_prod
    names["overall reconstructed"]=overall_reco
    names["ks+- share"]=total_reco
    names["ks00 share"]=total_reco_00
    names["signal side reconstructed"]= overall_sig_reco
    names["ks+- sig share"]=sig_reco
    names["ks00 sig share"]=sig_reco_00
    
    names["tag side reconstructed"]= tag_reco
    names["number of candidates +-"]=cand_list
    names["knumber of candidates k00"]=cand_list_00
    
    names["signal side reco Efficiency +-"]= eff
    names["signal side reco Purity +-"]= pur
    names["signal side reco Efficiency 00"]= eff_00
    names["signal side reco Purity 00"]= pur_00
    names["signal side reco Efficiency overall"]= overall_eff
    names["signal side reco Purity overall"]= overall_pur
    
    names["multiplicity +-"]= multi
    names["reconstructed events"]= evt
    names["multiplicity 00"]= multi_00
    names["reconstructed events 00"]= evt_00
    names["number of events"]= n
    
    
    
    ratio={"signal side reco Efficiency +-": str(sig_reco)+" / "+str(sig_prod * 0.692), \
           "signal side reco Purity +-": str(sig_reco)+" / "+str(cand_list)}
    
    def getratio(stat):
        try:
            return ratio[stat]
        except:
            return ""
        
    table=[]
    
    for stat in names:
        table.append([stat,names[stat],getratio(stat)])
    print("-"*10,tree,"-"*10)
    print (tabulate(table,numalign="right")) 
    print()
    print("KS mcMatching")
    error=t[t["K_S0_mcPDG"]==310]
    print(error.groupby("K_S0_mcErrors").size())
    print(len(t),len(error))
    
def getBg(df,tree):                         
    """
    Calculate the content of remaining Background events and produce a sorted list.
    """
    t = df[tree]    
    bg_reco = t[ t[getinfo(tree)["pdgstring"]]!=getinfo(tree)["pdg"]  ] 
    bg_sorted = pd.DataFrame(bg_reco.groupby([getinfo(tree)["pdgstring"]]).size().sort_values(ascending=False))
    bg_sorted.columns=["size"]
    
    bg_sorted["ratio"] = np.round(bg_sorted["size"]/bg_sorted["size"].sum(),3)
    bg_sorted["particle"]=""
    for row in bg_sorted.index.values:

        try: 
            bg_sorted.set_value(row,'particle', getmcstring(row))
             
        except:
            bg_sorted.set_value(row,'particle', "unknown")
            
    print("-"*10,tree,"Background study","-"*10)
    print(bg_sorted[:10])   
    print()

def getMcString(pdg):
    """
    Implement official PDG-code Values for Monte Carlo
    """
    mc={}
    mc[310]="Kdata"
    mc[300553]="Y(4S)"
    mc[511]="B0"
    mc[-511]="B0bar"
    mc[421]="D0"
    mc[-421]="D0bar"
    mc[411]="D+"
    mc[-411]="D-"
    
    return mc[pdg]
    
    
    
def getInfo(tree):
    """
    Map Tree-Variables into dictionary in order to seperate Values for Ks and B.
    """
  
    info={}
    if tree == "btree":
     info["pdgstring"] = "B0_mcPDG"
     info["pdg"]= 511
     info["truth"]= "btruthtree"
     info["motherstring"] ="B0_mcPDG"
     info["sigfactor"] = 1 
     info["truthfactor"] = 1 
     info["status"]=  'B0_mcErrors'
    elif tree == "kstree":
     info["pdgstring"] = "K_S0_mcPDG"
     info["pdg"]= 310
     info["truth"]= "kstruthtree"
     info["motherstring"] = 'K_S0_MC_MOTHER_ID'
     info["sigfactor"] = 3
     info["truthfactor"] = 0.5
     info["status"]=  'K_S0_mcErrors'
    elif tree == "ks00tree":
     info["pdgstring"] = "K_S0_mcPDG"
     info["pdg"]= 310
     info["truth"]= "kstruthtree"
     info["motherstring"] = 'K_S0_MC_MOTHER_ID'
     info["sigfactor"] = 3
     info["truthfactor"] = 0.5
     info["status"]=  'K_S0_mcErrors'
        
     return info

def gauss(x,A,mu,sigma):
        return  A * np.exp(-(x-mu)**2/(2.*sigma**2))

def triple(x,A1,mu1,sigma1,A2,mu2,sigma2,A3,mu3,sigma3):
        return gauss(x,A1,mu1,sigma1)+ gauss(x,A2,mu2,sigma2)+ gauss(x,A3,mu3,sigma3)


def deltaTFit(p,data):
    """ 
    Numeric convolution to the model is performed according to: \int [P_{sig}(\Delta t ') R_{sig}(\Delta t_i - \Delta t ')] d(\Delta t ').
    Determine the time dependent CPV-parameters S,A from the fit.
    """ 
  
    # set world average values for constants.
    deltaM = 0.507
    tauB0 = 1.519
    X = np.linspace(-20,20,100)
    p_i = 0.
    
    for x in X:
        p_i += (1 + data.B0_qrCombined * p[0]*np.sin(deltaM*x) \
                        + data.B0_qrCombined * p[1]*np.cos(deltaM*x))  \
           *np.exp(-(np.absolute(x)/tauB0))/(4 * tauB0)*triple((data.B0_DeltaT-x),*popt)   
    
    # build negative-logarithmic-likelihood
    return float(-np.log(p_i).sum())

      
def getUnc(array):
    """ 
    Calculate numeric 1 sigma uncertainty in order to feed it back to the BDToutput.
    """ 
  
  
    param = {str(sarray):"S_f",str(aarray):"A_f"}
    
    # Get Minimum of fit and the array's closest value to the +0.5 = 1 sigma value on each side of the parabel. 
    minimum = deltaTFit([S,A])
    onesigy = minimum+0.5
    slicer = np.argmin(abs(array - minimum))
    print(len(array),slicer,onesigy,minimum,min(array))
    downonesigxarr = np.argmin(abs(array[:slicer] - float(onesigy)))
    uponesigxarr = np.argmin(abs(array[slicer:] - float(onesigy))) + slicer
    
    # Map recieved values to x-axis and determine Uncertainty
    print(X[slicer],"slicer",slicer)
    print("upper: ",X[uponesigxarr],"lower: ",X[downonesigxarr])
    stat = (abs(X[uponesigxarr] - X[slicer]) + abs(X[downonesigxarr] - X[slicer])) /2. 
    statUp=  abs(X[uponesigxarr] - X[slicer])
    statDown=  abs(X[downonesigxarr] - X[slicer])
    print("stat",stat,statUp,statDown)
    
    # Determine actual Uncertainty from fit output according to model.
    sigmarealS = np.sqrt(2* (deltaTFit([-0.7,A]) - minimum))
    sigmarealA = np.sqrt(2* (deltaTFit([S,0]) - minimum))
    print(sigmarealS, " sigma")
    
    #Plot NegLogLike with 1 sigma lines.
    Xslice = X[(slicer-lowCut):(slicer+upCut)]
    arrayslice = array[(slicer-lowCut):(slicer+upCut)]
    plt.figure()
    plt.plot(Xslice, arrayslice,label=param[str(array)])
    plt.plot([X[downonesigxarr], X[uponesigxarr]],[onesigy, onesigy],'r--' ,label = r'$1 \sigma$')
    plt.axvline(x=X[uponesigxarr],  color='red', zorder=2, linestyle = '--')
    plt.axvline(x=X[downonesigxarr],  color='red', zorder=2, linestyle = '--')
    
    plt.legend()
    return (stat,minimum)
  
    