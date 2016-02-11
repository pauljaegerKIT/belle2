#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'pjaeger'

"""
Steering file commits jobs split up in small files 
in order to process 1 billion evnets.
For each background category mixed, charged, uubar, ddbar, ssbar, ccbar
files are sent.
If the job finishes succesfully the line "job done" is written into the log file.
"""

import os, sys, time ,  subprocess ,pickle

# collect directories for commit
dircs=[]

for i in range(1,13):
  dircs.append("mixed{}".format(i)

for i in range(1,13):
  dircs.append("charged{}".format(i)

for i in range(1,29):
  dircs.append("uubar{}".format(i)

for i in range(1,11):
  dircs.append("ddbar{}".format(i)

for i in range(1,10):
  dircs.append("ssbar{}".format(i)

for i in range(1,22):
  dircs.append("ccbar{}".format(i)



logdict={}

for dirc in dircs:
  loglist=[]
  indir = "/group/belle2/users/pjaeger/skimming/{}/".format(dirc)
  outdir = "/group/belle2/users/pjaeger/pretrain/{}/".format(dirc)
  print(indir,outdir)
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  
  
  for root, dirs, filenames in os.walk(indir):
    qcut = len(filenames)/4 
    queue = "s"
    count=0
    hadd=[]
    
    # If sending ~50k jobs the bottleneck is the submission itself.
    # Hence every 10 files are added to 1 for submission.
    
    for f in enumerate(filenames,start=0):
      # sort out log files
      if ".log" in f[0]:
        continue
      hadd.append(f[0])
      
      # Collect pack of 10 files and take care of rest of list.
      if len(hadd)==10 or f[1]==len(filenames):
        print("hadd {o} {i}".format(o=hadd[0],i=" ".join(i for i in hadd)))
        subprocess.call("hadd hadd_{o} {i}".format(o=hadd[0],i=" ".join(i for i in hadd)))
        hadd = []
	
	count +=1
	
	# Distribute jobs on different queues.
	if count>=qcut:
	  queue = "l"
	infile = indir+f "{id}/hadd_{i}".format(id=indir,i=hadd[0] )
	outfile = outdir+f "{id}/hadd_{i}".format(id=indir,i=hadd[0] )
      if os.path.exists("{}.log".format(outfile)):
        os.system("rm -rf {}.log".format(outfile))
      # send job.  
      subprocess.call("bsub -q {q}  -o {m}.log -e {n}.log basf2 pretrain_ks.py {i} {o} {od}".format(q= queue, m=outfile,n=outfile,i=infile, o=outfile, od=outdir), shell = True)
      loglist.append("{}.log".format(outfile))
  logdict[dirc] = loglist


# Pickle list of logfiles. Can be checked for completeness later.
if os.path.exists("logdictpre.p"):
  os.system("rm -rf logdictpre.p")
pickle.dump(logdict, open("logdictpre.p","wb"))

# Possible check for completeness and success of jobs performed every 20 seconds.
check=True
while check:
 check = False
 for key in logdict:
   if not os.path.exists(log) or not "job_done" in open(log).read():
     check = True
     time.sleep(20)
   else:
     break


"""
TODO: add outputs together to one file per category and perform Classifier Training. 
"""
