from pyscf import scf,gto,dft,cc
import numpy ,scipy, sys, os.path , re 
from scipy import linalg
from pyscf import lo 
from pyscf.lo.ibo import ibo
from pyscf.lo.edmiston import ER 

# Benjamin G. Janesko 
# July 2026 
# This standalone script performs post-Hartree-Fock Perdew-Zunger and
# local-hybrid-like self-interaction corrected LDA and PBE calculations.
# Molecule of interest is read in from a Gaussian formatted checkpoint file. 



names=['Dummy','H',          'He',
'Li','Be',   'B','C','N','O','F','Ne',
'Na','Mg',   'Al','Si','P','S','Cl','Ar',
'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe']
 
def readChk(file):

  # Read header information 
  NAO=0 
  NMO=0 
  Nat=0 
  b=''
  charge=0
  spin=0
  f = open(file,'r') 
  lines = f.readlines()
  b = lines[1].split().pop()
  if(b=='Gen'):
    b = 'def2qzvp'
  for l in lines:
    if('Number of atoms') in l:
      Nat= int(l.split().pop())
    if('Charge ') in l:
      charge= int(l.split().pop())
    if('Multiplicity ') in l:
     spin= int(l.split().pop())-1 
    if('Number of basis functions') in l:
      NAO = int(l.split().pop())
    if('Number of independent functions') in l:
      NMO = int(l.split().pop())
      break 
  print('Your file has basis ',b,' with ',Nat,' atoms and ',NAO,' ',NMO,' basis functions')
  print('Charge ',charge,' spin ',spin)
  print('Basis ',b)

  # Read lists of atom numbers and cartesin coordinates
  iats=[]
  cart=[]
  N = 0 
  r0= 0 
  rs= 0 
  for l in lines:
    if(len(iats)>=Nat):
      r0= 0 
    if(len(cart)>=N):
      rs= 0 
    if(r0>0):
      for x in l.split():
       if(len(iats)<Nat):
         iats.append(int(x))
    if(rs>0):
      for x in l.split():
       if(len(cart)<N):
         cart.append(float(x))
    if('Atomic numbers' in l):
      r0= 1 
    if('Current cartesian coordinates' in l):
      rs= 1 
      N =int(l.split().pop())

  # Repackage these into a PySCF molecule
  geom=''
  ind = -1 
  for iat in range(Nat):
    geom+= ' %4s ' %(names[iats[iat]])
    for i in range(3):
      ind = ind+1
      geom+=' %12.6f ' %(cart[ind])
    geom+='\n'

  #print('Your geometry is:\n ',geom)
  m = gto.Mole(atom=geom,charge=charge,spin=spin,basis=b)
  m.unit='B' # Gaussian uses Bohr units for geometries 
  #m.cart=True # Todo: Determine automatically 
  m.build() 
  NAO = m.nao 
  labs=m.ao_labels()

  # Read the basis functions in Gaussian order
  # PySCF reorders them as 
  # (1) atoms, (2) angular momentum, (3) shells, (4) spherical harmonics 
  ipy=[]
  for i in range(NAO):
    ipy.append(i)
  if(not m.cart):
    for i in range(NAO):
      if('dxy' in labs[i]): # Swap d subshells 
        ipy[i  ] = i+2
        ipy[i+1] = i+3
        ipy[i+2] = i+1
        ipy[i+3] = i+4
        ipy[i+4] = i+0
      if('f-3' in labs[i]): # Swap f subshells 
        ipy[i  ] = i+3
        ipy[i+1] = i+4
        ipy[i+2] = i+2
        ipy[i+3] = i+5
        ipy[i+4] = i+1
        ipy[i+5] = i+6
        ipy[i+6] = i+0
      if('g-4' in labs[i]): # Swap g subshells 
        ipy[i  ] = i+4
        ipy[i+1] = i+5
        ipy[i+2] = i+3
        ipy[i+3] = i+6
        ipy[i+4] = i+2
        ipy[i+5] = i+7
        ipy[i+6] = i+1
        ipy[i+7] = i+8
        ipy[i+8] = i+0

  # Read lists of total and spin density matrices 
  pdm0 = [] 
  pdms = [] 
  N = 0 
  r0= 0 
  rs= 0 
  for l in lines:
    if(len(pdm0)>=N):
      r0= 0 
    if(len(pdms)>=N):
      rs= 0 
    if(r0>0):
      for x in l.split():
       if(len(pdm0)<N):
         pdm0.append(float(x))
    if(rs>0):
      for x in l.split():
       if(len(pdms)<N):
         pdms.append(float(x))
    if('Total SCF Density' in l):
      r0= 1 
      N =int(l.split().pop())
    if('Spin SCF Density' in l):
      rs= 1 
  #print('You read in total 1PDM \n',pdm0,'\n and spin 1PDM\n',pdms)

  # Repackage these into a PySCF density matrix 
  P=numpy.zeros((2,NAO,NAO))
  ind = -1 
  for i in range(NAO):
    for j in range(i+1):
      ind = ind + 1 
      sp = 0 
      if(len(pdms)>0):
        sp = pdms[ind]
      v0 =  (pdm0[ind]+sp)/2
      v1 =  (pdm0[ind]-sp)/2
      P[0,ipy[i],ipy[j]] = v0
      P[0,ipy[j],ipy[i]] = v0
      P[1,ipy[i],ipy[j]] = v1
      P[1,ipy[j],ipy[i]] = v1

  # Read lists of alpha and beta orbital coefficients
  coefa= [] 
  coefb= [] 
  N = 0 
  ra= 0 
  rb= 0 
  for l in lines:
    if(len(coefa)>=N):
      ra= 0 
    if(len(coefb)>=N):
      rb= 0 
    if(ra>0):
      for x in l.split():
       if(len(coefa)<N):
         coefa.append(float(x))
    if(rb>0):
      for x in l.split():
       if(len(coefb)<N):
         coefb.append(float(x))
    if('Alpha MO coefficie' in l):
      ra= 1 
      N =int(l.split().pop())
    if('Beta MO coefficie' in l):
      rb= 1 
      N =int(l.split().pop())

  # Repackage these into a PySCF MO coefficient list [spin,ao,mo] 
  print('Coef len: ',len(coefa),len(coefb))
  mo_coeff=numpy.zeros((2,NAO,NMO))
  ind = -1 
  for imo in range(NMO):
    for iao in range(NAO):
      ind = ind + 1 
      v0 = coefa[ind]
      v1 = v0
      if(len(coefb)>0):
        v1 = coefb[ind]
      mo_coeff[0,ipy[iao],imo] = v0
      mo_coeff[1,ipy[iao],imo] = v1

  # Read lists of orbital energies
  aorb = [] 
  borb = [] 
  N = 0 
  ra= 0 
  rb= 0 
  for l in lines:
    if(len(aorb)>=N):
      ra= 0 
    if(len(borb)>=N):
      rb= 0 
    if(ra>0):
      for x in l.split():
       if(len(aorb)<N):
         aorb.append(float(x))
    if(rb>0):
      for x in l.split():
       if(len(borb)<N):
         borb.append(float(x))
    if('Alpha Orbital Energies' in l):
      ra= 1 
      N =int(l.split().pop())
    if('Beta Orbital Energies' in l):
      rb= 1 
  if(len(borb)<1):
    borb = aorb 
  print('Orbital array lengths ',len(aorb),len(borb))
  return(m,P,mo_coeff,aorb,borb)


fullbasis='def2tzvp'

if __name__=='__main__':
 tehFile=sys.argv[1]
 if(os.path.isfile(tehFile)):

   # Read the molecule 
   m,P,mo_coeff,aorb,borb = readChk(tehFile)
   fullbasis = m.basis 
   Na,Nb=m.nelec
   S = m.intor_symmetric('int1e_ovlp')
   Sm = numpy.linalg.inv(S)

   mf=scf.UHF(m)
   mf.kernel()
   P=mf.make_rdm1() 
   mo_coeff=mf.mo_coeff
   aorb=mf.mo_energy[0]
   borb=mf.mo_energy[1]

   Na2,Nb2=numpy.einsum('sij,ij->s',P,S)
   print('Molecule electrons ',Na,Nb,Na2,Nb2)
   if( (Na-Na2)**2 + (Nb-Nb2)**2 >0.0002):
     sys.exit('Your number of electrons is bad')
   NAO = m.nao
   print('NAO ',NAO)
   NMO = len(aorb)
   nat = m.natm
   ks=dft.UKS(m,xc='pbe,pbe')

   # Standard components of the total energy 
   h0=ks.get_hcore()
   Jmat2 = ks.get_j(dm=P)
   Jmat = Jmat2[0] + Jmat2[1] 
   Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
   K=ks.get_k(dm=P)
   EX=-0.5*numpy.einsum('sij,sij->',P,K)
   R=numpy.zeros_like(P)
   R[0]=-0.25*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
   R[1]=-0.25*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))

   # Edmiston-Reudenberg approximation to FLOSIC orbitals
   moas = mo_coeff[0,:,:Na]
   mobs = mo_coeff[1,:,:Nb]
   PPas=[]
   PPbs=[]
   RPas = [] 
   RPbs = [] 
   EXPas = [] 
   EXPbs = [] 
   if(moas.shape[1]<2):
     moasl = moas 
   else:
    mo3 = ibo(m,moas,locmethod='IBO',exponent=4)
    es = ER(m,mo3)
    es.kernel()
    moasl = es.mo_coeff[:,:Na] 
   if(mobs.shape[1]<2):
     mobsl = mobs 
   else:
    mo3 = ibo(m,mobs,locmethod='IBO',exponent=4)
    es = ER(m,mo3)
    es.kernel()
    mobsl = es.mo_coeff[:,:Nb] 
   for imo in range(Na):
    PP = numpy.zeros_like(P)
    PP[0] = numpy.outer(moasl[:,imo],moasl[:,imo])
    PPas.append(PP)
    KP=ks.get_k(dm=PP)
    EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
    EXPas.append(EXP)
    RP=-0.25*(numpy.dot(Sm,numpy.dot(KP[0],PP[0])) + numpy.dot(PP[0],numpy.dot(KP[0],Sm)))
    RPas.append(RP)
    test = numpy.dot(PP[0],numpy.dot(S,PP[0]))-PP[0] +  numpy.dot(PP[1],numpy.dot(S,PP[1]))-PP[1] 
   for imo in range(Nb):
    PP = numpy.zeros_like(P)
    PP[1] = numpy.outer(mobsl[:,imo],mobsl[:,imo])
    PPbs.append(PP)
    KP=ks.get_k(dm=PP)
    EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
    EXPbs.append(EXP)
    RP=-0.25*(numpy.dot(Sm,numpy.dot(KP[1],PP[1])) + numpy.dot(PP[1],numpy.dot(KP[1],Sm)))
    RPbs.append(RP)
    test = numpy.dot(PP[0],numpy.dot(S,PP[0]))-PP[0] +  numpy.dot(PP[1],numpy.dot(S,PP[1]))-PP[1] 

   print('SCF Done: E = %12.6f Hartree HF '%(Eother+EX))

   # Loop over XC functionals 
   for xc in ('lda,vwn','pbe,pbe',):
     x=xc.split(',')[0]
     x+=','
     EPZPas = [0] * Na
     EPZPbs = [0] * Nb
     ELHPas = [0] * Na
     ELHPbs = [0] * Nb
     ks=dft.UKS(m,xc=xc)
     ks.grids.level=8
  
     # Buffers for DFT integrated quantities 
     EX2=0
     ESL=0
     EPZ=0
     kvals=[1]
     ELPZ=[0]*len(kvals)
     ELLH=[0]*len(kvals)
     ELH=[0]*len(kvals)
     hermi=1 
     ni = ks._numint
     xctype=ni._xc_type(xc)
     xctype='MGGA' # So we can compute tauW/tau 
     ao_deriv=0
     if xctype == 'GGA':
       ao_deriv=1
     elif xctype == 'MGGA':
       ao_deriv=1
     nao=m.nao
     tiny = 0.00000000001
  
     # Makers for DFT densities, density gradients, etc.
     exctype = 'LDA' 
     make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
     make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
     make_exfa = ni._gen_rho_evaluator(m, [R[0]], hermi, False, ks.grids)[0]
     make_exfb = ni._gen_rho_evaluator(m, [R[1]], hermi, False, ks.grids)[0]
     make_rhopas=[]
     make_rhopbs=[]
     make_expas=[]
     make_expbs=[]
     for imo in range(Na):
      make_rhop= ni._gen_rho_evaluator(m, [PPas[imo][0]], hermi, False, ks.grids)[0]
      make_exp = ni._gen_rho_evaluator(m, [RPas[imo]], hermi, False, ks.grids)[0]
      make_rhopas.append(make_rhop)
      make_expas.append(make_exp)
     for imo in range(Nb):
      make_rhop= ni._gen_rho_evaluator(m, [PPbs[imo][1]], hermi, False, ks.grids)[0]
      make_exp = ni._gen_rho_evaluator(m, [RPbs[imo]], hermi, False, ks.grids)[0]
      make_rhopbs.append(make_rhop)
      make_expbs.append(make_exp)
  
     # Loop over batches 
     for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):

         # Exact exchange integrals 
         aosex=aos[0]
         if(ao_deriv<1):
           aosex = aos
   
         # Regular SL XC 
         rho_a = make_rhoa(0, aos, mask, xctype)
         nrho,ngrid = rho_a.shape 
         rho_b = make_rhob(0, aos, mask, xctype)
         if(len(rho_b.shape)>1):
           rho_b[0,rho_b[0]<tiny]=tiny
           rhob = rho_b[0]
         else:
           rho_b[rho_b<tiny]=tiny
           rhob = rho_b
         if(len(rho_a.shape)>1):
           rho_a[0,rho_a[0]<tiny]=tiny
           rhoa = rho_a[0]
         else:
           rho_a[rho_a<tiny]=tiny
           rhoa = rho_a
         rho = (rho_a, rho_b)

         excsl = ni.eval_xc_eff(xc, rho, deriv=0, xctype=exctype)[0]
         excsl = excsl*(rhoa+rhob) 
         exsl = ni.eval_xc_eff(x, rho, deriv=0, xctype=exctype)[0]
         exsl = exsl*(rhoa+rhob) 
         ESL = ESL + numpy.dot(excsl,weight)

         # Weight factors for spin-scaled LPZSIC and LLHSIC 
         gsqa = rho[0][1]**2+rho[0][2]**2+rho[0][3]**2
         gsqb = rho[1][1]**2+rho[1][2]**2+rho[1][3]**2
         ra = rho[0][0]
         ra = numpy.maximum(ra,1e-10*numpy.ones_like(ra))
         rb = rho[1][0]
         rb = numpy.maximum(rb,1e-10*numpy.ones_like(rb))
         fa = ra/(ra+rb)
         fb = rb/(ra+rb)
         ta = rho[0][4]
         ta = numpy.maximum(ta,1e-10*numpy.ones_like(ta))
         tb = rho[0][4]
         tb = numpy.maximum(tb,1e-10*numpy.ones_like(tb))
         twta = (gsqa/(8*ra*ta))**2 
         twtb = (gsqb/(8*rb*tb))**2 
   
         # Perdew-Zunger orbital density corrections 
         for imo in range(Na):
           expr=make_expas[imo](0, aosex, mask, exctype)
           rhop_a = make_rhopas[imo](0, aos, mask, xctype)
           if(len(rhop_a.shape)>1):
            rhop_a[0,rhop_a[0]<tiny]=tiny
            rhopa = rhop_a[0]
           else:
            rhop_a[rhop_a<tiny]=tiny
            rhopa = rhop_a
           rhop = (rhop_a, numpy.zeros_like(rhop_a))
           exc = ni.eval_xc_eff(x, rhop, deriv=0, xctype=xctype)[0]
           exc = exc*rhopa
           EPZPas[imo] = EPZPas[imo] + numpy.dot(exc,weight)
           for ik in range(len(kvals)):
             ELPZ[ik] = ELPZ[ik] + numpy.dot(weight,(twta*fa**kvals[ik])*(expr-exc))
           
         for imo in range(Nb):
           expr=make_expbs[imo](0, aosex, mask, exctype)
           rhop_b = make_rhopbs[imo](0, aos, mask, xctype)
           if(len(rhop_b.shape)>1):
            rhop_b[0,rhop_b[0]<tiny]=tiny
            rhopb = rhop_b[0]
           else:
            rhop_b[rhop_b<tiny]=tiny
            rhopb = rhop_b
           rhop = (numpy.zeros_like(rhop_b), rhop_b)
           exc = ni.eval_xc_eff(x, rhop, deriv=0, xctype=xctype)[0]
           exc = exc*rhopb
           EPZPbs[imo] = EPZPbs[imo] + numpy.dot(exc,weight)
           for ik in range(len(kvals)):
             ELPZ[ik] = ELPZ[ik] + numpy.dot(weight,(twtb*fb**kvals[ik])*(expr-exc))
  
         # Local hybrid type corrections exchange only 
         exa = make_exfa(0, aosex, mask, exctype)
         exb = make_exfb(0,aosex,mask,exctype)
         ex=exa+exb
         ex[ex>-tiny]=-tiny
         EX2 = EX2 + numpy.dot(ex,weight)
         exsla = exsl*ra/(ra+rb)
         exslb = exsl*rb/(ra+rb)
         for ik in range(len(kvals)):
           ELH[ik] = ELH[ik] + numpy.dot(weight,(twta*fa**kvals[ik])*(exa-exsla))
           ELH[ik] = ELH[ik] + numpy.dot(weight,(twtb*fb**kvals[ik])*(exb-exslb))
         for imo in range(Na):
           expr=make_expas[imo](0, aosex, mask, exctype)
           fac = expr/ex
           fac[fac>1.2]=1.2
           fac[fac<-1.2]=-1.2
           exclh = exsl*fac
           ELHPas[imo] = ELHPas[imo] + numpy.dot(exclh,weight)
           for ik in range(len(kvals)):
             ELLH[ik] = ELLH[ik] + numpy.dot(weight,(twta*fa**kvals[ik])*(expr-exclh))
         for imo in range(Nb):
           expr=make_expbs[imo](0, aosex, mask, exctype)
           fac = expr/ex
           fac[fac>2]=2
           fac[fac<-2]=-2
           exclh = exsl*fac
           ELHPbs[imo] = ELHPbs[imo] + numpy.dot(exclh,weight)
           for ik in range(len(kvals)):
             ELLH[ik] = ELLH[ik] + numpy.dot(weight,(twtb*fb**kvals[ik])*(expr-exclh))
  
     # Print results 
     EXPall = sum(EXPas) + sum(EXPbs)
     EPZall = ESL + EXPall - (sum(EPZPas)+sum(EPZPbs))
     ELHall = ESL + EXPall - (sum(ELHPas)+sum(ELHPbs))
     print('RES %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f '%(EX,EX2,EXPall,ESL,EPZall,ELHall))
     print('SCF Done: E = %12.6f Hartree DFT %s '%(Eother+ESL,xc))
     print('SCF Done: E = %12.6f Hartree DFT-PZ %s '%(Eother+EPZall,xc))
     print('SCF Done: E = %12.6f Hartree DFT-LH %s '%(Eother+ELHall,xc))
     for ik in range(len(kvals)):
       print('SCF Done: E = %12.6f Hartree DFT-LH  %.2f %s '%(Eother+ESL+ELH[ik],kvals[ik],xc))
     for ik in range(len(kvals)):
       print('SCF Done: E = %12.6f Hartree DFT-LPZ %.2f %s '%(Eother+ESL+ELPZ[ik],kvals[ik],xc))
     for ik in range(len(kvals)):
       print('SCF Done: E = %12.6f Hartree DFT-LLH %.2f %s '%(Eother+ESL+ELLH[ik],kvals[ik],xc))
