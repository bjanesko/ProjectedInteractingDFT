#!/usr/bin/env python
# Work routines for projected DFT 
import time
import math 
import numpy, sys 
from scipy import linalg
from scipy.special import erf 
from pyscf import gto ,ao2mo, mcscf , lib , scf, cc 
from pyscf.mp import ump2
from dfrdump2_native import DFURDMP2
from pyscf.dft.gen_grid import BLKSIZE, NBINS, CUTOFF, make_mask
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse

# Return a copy of input molecule m with stretched AOs
def stretchAOs(m,beta):
  if m.cart:
     raise ValueError('Cartesian AOs not implemented with r35')
  if not m._built:
        logger.warn(m, 'Warning: stretchAOs object %s not initialized. Initializing %s',
                    m, m)
        m.build()
  m2 = m.copy() 
  m2.build()
  nshell = (m2._bas.shape)[0]
  _oldenv = m._env.copy()
  for ishell in range(nshell):
    lshell = m2._bas[ishell,1]
    nprim = m2._bas[ishell,2]
    expstart = m2._bas[ishell,5]
    constart = m2._bas[ishell,6]
  
    # Rescale the exponents and contraction coefficients for each shell
    # Note this fails if different shells share the same exponents, be careful 
    for iexp in range(nprim):
      astart = _oldenv[expstart+iexp]
      cstart = _oldenv[constart+iexp]
      #print('+++ basis exponent and coeff',astart,cstart)
      aend = astart*beta/(astart+beta) # Stretched exponent 
      cend = cstart*(2*math.pi*beta)**0.75/(astart+beta)**1.5  # Renormalization 
      cend = cend*(beta/(astart+beta))**lshell # Shell dependence 
      #print('Rescaling shell ',ishell,' exponent ',astart,' to ',aend)
      m2._env[expstart+iexp] = aend 
      m2._env[constart+iexp] = cend 
  return(m2) 


def get_d(m):
  DAOs = [] 

  # Find the transition metal atoms 
  tms=[]
  for iat in range(m.natm):
    iq = m.atom_charge(iat)
    if( (iq>=21 and iq<=30) or (iq>=39 and iq<48) ):
      tms.append(iat)
  ntm = len(tms)
  print(' We have ',ntm,' transition metal atoms')

  # Each transition metal has five sets of d orbitals 
  labs = m.ao_labels()
  for iat in tms:
     for ityp in ('dxy','dyz','dz^2','dxz','dx2-y2'):
        vals=[]
        for iao in range(m.nao):
          icen = int(labs[iao].split()[0])
          #print('Comparing label ',labs[iao],' to type ',ityp)
          if((icen==iat) and (ityp in labs[iao])):
            DAOs.append(iao)
            vals.append(iao)
  DAOs = [*set(DAOs)] # Remove duplicates 
  DAOs = [DAOs] # one set  
  return(DAOs)

def assign_cores(m):
  # Maximum valence and minimum core exponent for atoms H-Ar, STO-2G basis set 
  MaxValence=[0, 0, 0.246, 0.508, 0.864, 1.136, 1.461, 1.945, 2.498, 3.187, 0.594, 0.561, 0.561, 0.594, 0.7, 0.815, 0.855, 1.053]
  MinCore=[0, 0, 1.097, 2.053, 3.321, 4.874, 6.745, 8.896, 11.344, 14.09, 1.18, 1.482, 1.852, 2.273, 2.747, 3.267, 3.819, 4.427]

  # Compute each AO's radial extent in terms of kinetic energy integrals 
  CoreAOs = [] 
  T=m.intor_symmetric('int1e_kin')
  labs = m.ao_labels()
  for iao in range(m.nao):
    icen = int(labs[iao].split()[0])
    iat = m.atom_charge(icen)
    acut = MinCore[iat-1]
    Tcut = acut*1.5 * 1.0
    Tval = T[iao,iao]
    #print("AO ",iao," ",labs[iao]," center ",icen," atom charge ",iat," T ",Tval," Tcut ",Tcut)
    if(Tval>Tcut and iat>2 and ('s ' in labs[iao] )):
      CoreAOs.append(iao)
      #print(labs[iao])

  CoreAOs = [*set(CoreAOs)] # Remove duplicates 
  return(CoreAOs)

def mo_proj(ks):
    ''' Build a single projection operator from an input list of orthonormal MOs 
    Args:
        ks : an instance of :class:`RKS`
    '''  
    orbs = ks.paos[0]
    m = ks.mol
    N = m.nao
    S = ks.get_ovlp() 
    (NAO,NC) = orbs.shape
    #print('N ',N)
    #print('Orbs: ',orbs.shape)
    if(NAO != N):
      die('Failure in mo_proj with ',N,' and ',orbs.shape)
    Q =numpy.zeros((N,N))
    for i in range(NC):
      v = orbs[:,i]
      Q = Q + numpy.outer(v,v)
    ks.QS=[numpy.einsum('ik,kj->ij',Q,S)]
    ks.SQ=[numpy.einsum('ik,kj->ij',S,Q)]
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))
   
gsspins=[1,0, 
1,0,  1,2,3,2,1,0, 
1,0,  1,2,3,2,1,0,  
1,0, 1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0,   
1,0, 1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0] 
valencetag=['1s','1s',
'2s','2s',  '2p','2p','2p','2p','2p','2p',
'3s','3s',  '3p','3p','3p','3p','3p','3p',
'4s','4s',  '3d','3d','3d','3d','3d','3d','3d','3d','3d','3d',   '4p','4p','4p','4p','4p','4p',
'5s','5s',  '4d','4d','4d','4d','4d','4d','4d','4d','4d','4d',   '5p','5p','5p','5p','5p','5p'] 

def build_mbproj_2(ks):
  # June 2024, this version builds a set of orthogonalized projected atomic
  # orbitals (opAOs) on each atom, from the STO-3G minimal basis, then builds projector
  # Q from orthogonalizing all of those. The opAOs are saved in ks.proj for use
  # in euci. This sets up pAOs in current basis from projection onto minimal
  # basis, then calls pao_proj 
  # Note this requires PROJECTING the minimal basis onto the current AO basis, thus it requires Sinv 
  # It also requires a tag for the 'valence' AOs of each atom 
  # July 2025 make sure VeepAOs is saved. August 2025 make sure VeepAOs includes range separation 

  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  m = ks.mol
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  print('BUILD_MBPROJ_2 OMEGA ALPHA HYB ',omega,alpha,hyb)
  thepaos= [] 
  theVeepAOs= [] 
  theVeeRSpAOs= []  # Range separation 
  print('Projected minimal AOs in build_mbproj_2')
  #basis = 'sto-3g'
  basis = m.basis # this aids recovery of FCI and avoids Eproj<E
  mmin =gto.Mole(atom=m.atom,basis=basis,charge=m.charge,spin=m.spin)
  mmin.unit=m.unit
  mmin.build()
  SX = gto.intor_cross('int1e_ovlp',m,mmin)
  labs = mmin.ao_labels()
  #print('Searching in labels \n',labs)
  for iat in range(m.natm):
    thisatpaos=[] 
    attype = m.atom_charge(iat)-1 
    tag = valencetag[attype]
    keptaos=[]
    for iao in range(mmin.nao):
      icen = int(labs[iao].split()[0])
      if(tag in labs[iao] or True): # TEST with all 
        if(icen == iat): 
          print(iat,': ',labs[iao])
          keptaos.append(iao)
          thisatpaos.append(numpy.dot(Sm,SX[:,iao]))
    thisatpaos2=numpy.transpose(numpy.array(thisatpaos))
    thepaos.append(thisatpaos2)

    # September 2024, use this atom's minimal basis twoelecints as the pAO
    # twoelecints . August 2025 add range separation 
    matmin=gto.Mole(atom=m.atom_symbol(iat),basis=basis,charge=0,spin=gsspins[attype])
    matmin.build() 
    keptataos=[]
    atlabs = matmin.ao_labels()
    for iao in range(matmin.nao):
      if(tag in atlabs[iao] or  True): # TEST with all 
        keptataos.append(iao)
    eri0=matmin.intor('int2e')
    eri0=eri0.reshape(matmin.nao,matmin.nao,matmin.nao,matmin.nao)
    matmin.set_range_coulomb(omega) # Range separation 
    eri1=matmin.intor('int2e')
    eri1=eri1.reshape(matmin.nao,matmin.nao,matmin.nao,matmin.nao)
    npao = len(keptataos)
    VeepAO=numpy.zeros((npao,npao,npao,npao))
    VeeRSpAO=numpy.zeros((npao,npao,npao,npao))
    print('TEST SIZES ',npao,matmin.nao,VeepAO.shape,eri0.shape)
    for i in range(npao):
      for j in range(npao):
        for k in range(npao):
          for l in range(npao):
            VeepAO[i,j,k,l]=eri0[keptataos[i],keptataos[j],keptataos[k],keptataos[l]]
            VeeRSpAO[i,j,k,l]=eri1[keptataos[i],keptataos[j],keptataos[k],keptataos[l]]
    theVeepAOs.append(VeepAO)
    theVeeRSpAOs.append(VeeRSpAO)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  ks.VeeRSpAOs= theVeeRSpAOs
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  #print('LOOK HERE IS KS.QS[0] ',ks.QS[0])
  return

def build_mbproj_spin(ks):
  # Feb 2026, this builds a set of orthogonalized projected fragment 
  # orbitals from a list of imput delocalized orbitals
  # Input is ks.fragments [
  # [definition of fragments as atom lists], 
  # [number of paos in each fragment], 
  # delocalized spinorbs used to build fragments
  if(not (hasattr(ks,'fragments'))):
    sys.exit('No fragments defined in ks for build_mbproj_spin')
  m = ks.mol
  nao = m.nao
  fragats,fragnums,spinorbs=ks.fragments
  nfrag =len(fragnums)
  if(nfrag != len(fragats)):
    sys.exit('nfrag %d and fragats %d in build_mbproj_spin'%(nfrag,len(fragats)))
  nspinorb=spinorbs.shape[1] # AO, MO 
  if(spinorbs.shape[0]!=nao): 
    sys.exit('spinorbs nao %d and mol nao %d in build_mbproj_spin'%(spinorbs.shape[0],nao))
  S = ks.get_ovlp() 

  # Assign each AO to a fragment 
  atom_of_ao = [0] * nao
  iao = 0
  for ishell in range(m.nbas):
    l = m.bas_angular(ishell)
    nc = m.bas_nctr(ishell)
    if(m.cart):
      nb = (l+1)*(l+2)/2*nc
    else:
      nb = (2*l+1)*nc
    iat = m.bas_atom(ishell)
    atom_of_ao[iao:iao+nb]=[iat]*nb
    iao = iao+nb
  fragment_of_ao = [-1]*nao
  aos_in_frag = []
  for ifrag in range(nfrag):
    aos_in_frag.append([])
  for iao in range(nao):
     iat = atom_of_ao[iao]
     for ifrag in range(nfrag):
       if iat in fragats[ifrag]:
         fragment_of_ao[iao] = ifrag 
         aos_in_frag[ifrag].append(iao)

  # Final loop over fragments 
  thepaos = [] 
  theVeepAOs= [] 
  for ifrag in range(nfrag):
    npao= fragnums[ifrag] # We'll generate npao paos for this fragment 

    # Build a fragment molecule for building Vee
    fraggeom =''
    for iat2 in fragats[ifrag]:
      xyz = m.atom_coords()[iat2]
      fraggeom = fraggeom + '%4s %12.6f %12.6f %12.6f \n'%(m.atom_symbol(iat2),xyz[0],xyz[1],xyz[2])
    mfrag = gto.Mole(atom=fraggeom,basis=m.basis,charge=0)
    mfrag.unit=m.unit 
    mfrag.cart=m.cart
    if(mfrag.nelectron % 2 == 1):
      mfrag.spin=1
    mfrag.build() 
    Sfrag= mfrag.intor_symmetric('int1e_ovlp')
    print('Fragment %d contains atoms '%(ifrag))
    print(fragats[ifrag])
    print(' and %d aos %d projected orbs'%(mfrag.nao,npao))

    # Find each spinorb's projection onto this fragment. 
    # Save the projections and the renormalized spinorbitals 
    spinnorms=[]
    spinorbfrags=[]
    print('Fragment ',ifrag,' S shape ',Sfrag.shape)
    for ispin in range(nspinorb):
      spinorbfrag = [spinorbs[iao,ispin] for iao in aos_in_frag[ifrag]]
      spinorbfrag = numpy.array(spinorbfrag)
      nsf = numpy.dot(spinorbfrag,numpy.dot(Sfrag,spinorbfrag))
      spinnorms.append(nsf)
      spinorbfrags.append(spinorbfrag*nsf**(-0.5))

    print('Fragment ',ifrag,' spinorb projections \n',spinnorms)

    # Select the spinorbs used to build phi for this fragment, in the fragment basis 
    ssn = sorted(range(nspinorb),key=spinnorms.__getitem__,reverse=True)
    thisfragorbs = []
    for i in range(npao):
      thisfragorbs.append(spinorbfrags[ssn[i]])
      print('Fragment %d gets spinorb %d projection %.3f '%(ifrag,ssn[i],spinnorms[ssn[i]]))
    thisfragorbs = numpy.array(thisfragorbs)
    print('Fragment ',ifrag,' phi overlap, frag \n',numpy.einsum('im,mn,jn->ij',thisfragorbs,Sfrag,thisfragorbs))

    # Generate Vee in fragment basis and phi in the full basis 
    tfo2 = numpy.zeros((mfrag.nao,npao)) # AO, MO 
    thisfragorbs2 = numpy.zeros((nao,npao)) # AO, MO
    for ipao in range(npao):
       tfo2[:,ipao]=thisfragorbs[ipao]
       for i in range(len(aos_in_frag[ifrag])):
          iao = aos_in_frag[ifrag][i]
          thisfragorbs2[iao,ipao] = thisfragorbs[ipao,i]
    print('Fragment ',ifrag,' phi overlap, full \n',numpy.einsum('mi,mn,nj->ij',thisfragorbs2,S,thisfragorbs2))
    VeepAO2 = mfrag.ao2mo(mo_coeffs=tfo2)
    VeepAO = ao2mo.restore(1,numpy.asarray(VeepAO2),tfo2.shape[1])
    theVeepAOs.append(VeepAO)
    thepaos.append(thisfragorbs2)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  ks.VeeRSpAOs= theVeepAOs 
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  return

def build_mbproj_fragment(ks):
  # October 2025, this builds a set of orthogonalized projected fragment 
  # orbitals (opFOs) on each fragment, from the STO-3G minimal basis, then builds projector
  # Q from orthogonalizing all of those. The opFOs are saved in ks.proj for use
  # in euci. Also saves VeepFOs from the fragment Vee for euci.
  # 
  # Input is ks.fragments, a tuple of [ 0-start start atom in the fragment,[projected
  # fragment orbs for the fragment]]. The atom lists must be contiguous and the
  # projected fragment orbs must have the same atom ordering as the atom list
  if(not (hasattr(ks,'fragments'))):
    sys.exit('No fragments defined in ks for build_mbproj_fragment')
  fragstarts,fragorbs,fragbasis =ks.fragments
  if(fragbasis is None):
    fragbasis = 'sto-3g'
  nfrag = len(fragstarts)
  print('Fragment basis ',fragbasis)
  print('Fragment starts ',fragstarts)
  #print('Fragment orbs ',fragorbs)
  
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  m = ks.mol
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  print('BUILD_MBPROJ_FRAGMENT OMEGA ALPHA HYB ',omega,alpha,hyb)
  thepaos= [] 
  theVeepAOs= [] 
  theVeeRSpAOs= []  # Range separation 
  print('Projected minimal AOs in build_mbproj_fragment')

  # Full molecule in fragment basis 
  mmin =gto.Mole(atom=m.atom,basis=fragbasis,charge=m.charge,spin=m.spin)
  mmin.unit=m.unit
  mmin.cart=m.cart
  mmin.build()
  SX = gto.intor_cross('int1e_ovlp',m,mmin)

  # Fragment molecules in fragment basis and fragment indexing 
  iaostart = 0
  for iat in range(nfrag):
    atstart=fragstarts[iat]
    atend = mmin.natm
    if(iat<nfrag-1):
      atend = fragstarts[iat+1]
    thisfragorbs = fragorbs[iat]
    #print('Look here is the shape of fragment ',iat,' orbitals ',thisfragorbs.shape)
    npao = thisfragorbs.shape[0] # len(thisfragorbs)

    fraggeom =''
    for iat2 in range(atstart,atend):
      xyz = m.atom_coords()[iat2]
      fraggeom = fraggeom + '%4s %12.6f %12.6f %12.6f \n'%(m.atom_symbol(iat2),xyz[0],xyz[1],xyz[2])
    mfrag = gto.Mole(atom=fraggeom,basis=fragbasis,charge=0)
    mfrag.cart=m.cart
    if(mfrag.nelectron % 2 == 1):
      mfrag.spin=1
    mfrag.build() 
    print('Fragment %d contains atoms %d-%d and %d aos %d projected orbs'%(iat,atstart,atend,mfrag.nao,npao))

    tfo2 = numpy.zeros((mfrag.nao,npao)) # spin, AO, MO 
    for ipao in range(npao):
        tfo2[:,ipao]=thisfragorbs[ipao]
    print('tfo2 shape ',tfo2.shape)

    Sfrag= mfrag.intor_symmetric('int1e_ovlp')

    #eri0=mfrag.intor('int2e').reshape(mfrag.nao,mfrag.nao,mfrag.nao,mfrag.nao)
    #mfrag.set_range_coulomb(omega) # Range separation 
    #eri1=mfrag.intor('int2e').reshape(mfrag.nao,mfrag.nao,mfrag.nao,mfrag.nao)
    #t1 = numpy.einsum('mnpq,lq->mnpl',eri0,thisfragorbs)
    #t2 = numpy.einsum('mnpl,kp->mnkl',t1,thisfragorbs)
    #t1 = numpy.einsum('mnkl,jn->mjkl',t2,thisfragorbs)
    #VeepAO= numpy.einsum('mjkl,im->ijkl',t1,thisfragorbs)
    #t1 = numpy.einsum('mnpq,lq->mnpl',eri1,thisfragorbs)
    #t2 = numpy.einsum('mnpl,kp->mnkl',t1,thisfragorbs)
    #t1 = numpy.einsum('mnkl,jn->mjkl',t2,thisfragorbs)
    #VeeRSpAO= numpy.einsum('mjkl,im->ijkl',t1,thisfragorbs)
    print('LOOK WE HAVE MO COEFFS SHAPE ',tfo2.shape)
    VeepAO2 = mfrag.ao2mo(mo_coeffs=tfo2)
    VeepAO = ao2mo.restore(1,numpy.asarray(VeepAO2),tfo2.shape[1])
    print('VEES SHAPE ',VeepAO.shape)
    print('VEES FIRST ',VeepAO[0,0,0,0])
    mfrag.set_range_coulomb(omega) # Range separation 
    VeeRSpAO2 = mfrag.ao2mo(mo_coeffs=tfo2)
    VeeRSpAO= ao2mo.restore(1,numpy.asarray(VeeRSpAO2),tfo2.shape[1])
    theVeepAOs.append(VeepAO)
    theVeeRSpAOs.append(VeeRSpAO)


    # Expand out fragment orbitals into full molecule basis set 
    thisfragorbs2 = numpy.zeros((npao,mmin.nao))
    for i in range(npao):
      thisfragorbs2[i,iaostart:iaostart+mfrag.nao] = thisfragorbs[i]
    iaostart = iaostart + mfrag.nao 

    thisatpaos=[] 
    for iao in range(npao):
       fragorb = thisfragorbs2[iao]
       print('Fragment %d orb %d ovlp %12.6f \n'%(iat,iao,numpy.dot(fragorb,numpy.dot(S,fragorb))))
       #print('Teh sizes ',Sm.shape,SX.shape,fragorb.shape)
       thisatpaos.append(numpy.dot(Sm,numpy.dot(SX,fragorb)))
    #iao = iao+mfrag.nao
    thisatpaos2=numpy.transpose(numpy.array(thisatpaos))
    thepaos.append(thisatpaos2)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  ks.VeeRSpAOs= theVeeRSpAOs
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  return
       
def build_mbproj_3(ks):
  # August 2024, this version builds a set of orthogonalized projected atomic
  # orbitals (opAOs) on each atom, from the 3-21g basis, then builds projector
  # Q from orthogonalizing all of those. The opAOs are saved in ks.proj for use
  # in euci. This sets up pAOs in current basis from projection onto minimal
  # basis, then calls pao_proj 
  # Note this requires PROJECTING the minimal basis onto the current AO basis, thus it requires Sinv 
  # It also requires a tag for the 'valence' AOs of each atom 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  m = ks.mol
  thepaos= [] 
  theVeepAOs= [] 
  print('Projected AOs in build_mbproj_3')
  mmin =gto.Mole(atom=m.atom,basis='3-21g',charge=m.charge,spin=m.spin)
  mmin.unit=m.unit
  mmin.build()
  SX = gto.intor_cross('int1e_ovlp',m,mmin)
  labs = mmin.ao_labels()
  for iat in range(m.natm):
    thisatpaos=[] 
    attype = m.atom_charge(iat)
    for iao in range(mmin.nao):
      icen = int(labs[iao].split()[0])
      if(icen == iat): 
        keep=1
        #if(attype>2 and ('1s' in labs[iao])):
        #  keep=0
        #if(attype>10 and ('2s' in labs[iao] or '2p' in labs[iao])):
        #  keep=0
        #if(attype>18 and ('3s' in labs[iao] or '3p' in labs[iao])):
        #  keep=0
        if(keep):
          print(iat,': ',labs[iao])
          thisatpaos.append(numpy.dot(Sm,SX[:,iao]))
    thisatpaos2=numpy.transpose(numpy.array(thisatpaos))
    thepaos.append(thisatpaos2)

    # December 2024 use this atom's 3-21g basis twoelecints as the pAO
    # twoelecints to accelerate 
    attype = m.atom_charge(iat) 
    print('Getting twoelecints for atom ',attype,gsspins[attype])
    matmin=gto.Mole(atom=m.atom_symbol(iat),basis='3-21g',charge=0,spin=gsspins[attype-1])
    matmin.build() 
    keptataos=[]
    atlabs = matmin.ao_labels()
    for iao in range(matmin.nao):
        keep=1
        #if(attype>2 and ('1s' in atlabs[iao])):
        #  keep=0
        #if(attype>10 and ('2s' in atlabs[iao] or '2p' in atlabs[iao])):
        #  keep=0
        #if(attype>18 and ('3s' in atlabs[iao] or '3p' in atlabs[iao])):
        #  keep=0
        if(keep):
          print(atlabs[iao])
          keptataos.append(iao)
      #if(tag in atlabs[iao]):
      #  keptataos.append(iao)
    eri0=matmin.intor('int2e')
    eri0=eri0.reshape(matmin.nao,matmin.nao,matmin.nao,matmin.nao)
    npao = len(keptataos)
    VeepAO=numpy.zeros((npao,npao,npao,npao))
    print('TEST SIZES ',npao,matmin.nao,VeepAO.shape,eri0.shape)
    for i in range(npao):
      for j in range(npao):
        for k in range(npao):
          for l in range(npao):
            VeepAO[i,j,k,l]=eri0[keptataos[i],keptataos[j],keptataos[k],keptataos[l]]
    theVeepAOs.append(VeepAO)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  return
       
  

def build_mbproj(ks,daos=False,faos=False,vaos=False,dum=False):
    ''' Build projection operators from an existing AO basis 
    This version builds two projection operators: one for second-period
    atoms Li-Ne, one for third-period atoms Na-Ar. 
    With daos=True, we get one projection operator for transition metal atoms 
    With vaos=True, we get one projection operator containing minimal basis AOs in this basis 
    Args:
        ks : an instance of :class:`RKS`
    June 2024, modify paos to include a list of matrices of pAOs from each shell 
    '''  
    if(ks.QS is None):
      m = ks.mol
      S = ks.get_ovlp() 
      Sm = linalg.inv(S)
      N=(S.shape)[0]
      Q2 = numpy.zeros((N,N))
      Q3 = numpy.zeros((N,N))
      ks.QS=[]
      ks.SQ=[] 

      # Prepare dummy molecule with minimal basis set 
      md = m.copy()
      md.basis='sto3g'   
      md.build() 
      SM = md.intor_symmetric('int1e_ovlp')
      
      # Build cross-overlap matrix between current and minimal core 
      SX = gto.intor_cross('int1e_ovlp',m,md)
      
      # Find minimal basis AOs used for this projection 
      coreaos=[]
      coreassign=[]
      coreatind=[]
      labs = md.ao_labels()
      for iao in range(md.nao):
        icen = int(labs[iao].split()[0])
        iat = md.atom_charge(icen) + md.atom_nelec_core(icen)
        if(daos):
          if( ((iat>20 and iat<31) or (iat>38 and iat<49)) and   ('d' in labs[iao]) ):
            coreaos.append(iao)
            coreassign.append(2)
            print(labs[iao],coreassign[-1])
        elif(faos):
          if( ((iat>56 and iat<72) or (iat>88 and iat<104)) and   ('f' in labs[iao]) ):
            coreaos.append(iao)
            coreassign.append(2)
            print('Proj AO',labs[iao],coreassign[-1])
        elif(vaos):
          if( ((iat<3) and   ('1s' in labs[iao])) 
or ((iat>2 and iat<5) and  ('2s' in labs[iao])) 
or ((iat>4 and iat<11) and  ('2p' in labs[iao])) 
or ((iat>10 and iat<13) and  ('3s' in labs[iao])) 
or ((iat>12 and iat<19) and  ('3p' in labs[iao])) 
or ((iat>18 and iat<21) and  ('4s' in labs[iao])) 
or ((iat>20 and iat<31) and  ('3d' in labs[iao])) 
or ((iat>20 and iat<31) and  ('4s' in labs[iao])) 
or ((iat>30 and iat<37) and  ('4p' in labs[iao])) 
or ((iat>36 and iat<39) and  ('5s' in labs[iao])) 
or ((iat>48 and iat<55) and  ('5p' in labs[iao])) 
):
            coreaos.append(iao)
            coreatind.append(icen) # BGJ label each AO with the center index 
            coreassign.append(2) 
            print('Proj AO',labs[iao],coreassign[-1])
        elif(dum):
          #print('+++ ',iao,labs[iao])
          if(iat>2 and   (('py' in labs[iao]) or ('pz' in labs[iao]) or ('px' in labs[iao])) ):
            coreaos.append(iao)
            coreassign.append(2)
            print(labs[iao],coreassign[-1])
        else:
          #print('YOU ARE THERE WITH IAT ',iat,' daos ',daos,' faos ',faos)
          if(iat>2 and   (' 1s' in labs[iao]) ):
             coreaos.append(iao)
             if(iat>10):
               coreassign.append(3)
             else: 
               coreassign.append(2)
             print(labs[iao],coreassign[-1])
          if(iat>10 and   (' 2s' in labs[iao] or ' 2p' in labs[iao]) ):
             coreaos.append(iao)
             coreassign.append(3)
             print(labs[iao],coreassign[-1])

      # Project kept core AOs into this basis
      NC = len(coreaos)
      if(NC>0):
        SC = numpy.zeros((NC,NC))
        SXC = numpy.zeros((N,NC))
        for ic in range(NC):
          SXC[:,ic] = SX[:,coreaos[ic]]
          for jc in range(NC):
             SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]

        # New June 2024 put projected core AOs from each atom into paos 
        # paos will be a list of Nsite rectangular matrices
        # Does not work! 
        if(vaos):
          projats = numpy.unique(coreatind)
          paosthis = [] 
          for iat in range(len(projats)):
            thisatpaos=[]
            for ic in range(NC):
              if(coreatind[ic] == projats[iat]):
                thisatpaos.append(SXC[:,ic])
          thisatpaos2 = numpy.transpose(numpy.array(thisatpaos))
          print('TEST SHAPE ',thisatpaos2.shape)
          paosthis.append(thisatpaos2)
          ks.paos = paosthis

        # Prepare two SC^(-1) operators, one for 2nd period elements and one for 3rd period elements 
        # Assume that the ith orthogonalized core AO equals the ith core AO 
        #SCm= linalg.inv(SC)
        (vals,vecs) = linalg.eigh(SC)
        SCm2 = numpy.zeros((NC,NC))
        SCm3 = numpy.zeros((NC,NC))
        SCmhalf = numpy.zeros((NC,NC))
        for i in range(NC):
          if(vals[i]>0.00000001):
            SCmhalf[i,i] = ((vals[i]).real)**(-0.5)
          else:
            print('Eliminating overlap eigenvalue ',i,vals[i])
        QCbig = numpy.dot(vecs,numpy.dot(SCmhalf,numpy.transpose(vecs)))
        for ic in range(NC):
          v = QCbig[ic]
          Qset = numpy.outer(v,v)
          if(coreassign[ic] == 3):
            SCm3 = SCm3 + Qset 
          else:
            SCm2 = SCm2 + Qset 
      
        # Core AO projection operators in current basis set 
        #Q = numpy.einsum('ia,ab,bc,dc,dj->ij',Sm,SXC,SCm,SXC,Sm)
        SmSXC= numpy.dot(Sm,SXC)
        t2 = numpy.dot(SmSXC,SCm2)
        Q2 = numpy.dot(t2,numpy.transpose(SmSXC))
        t2 = numpy.dot(SmSXC,SCm3)
        Q3 = numpy.dot(t2,numpy.transpose(SmSXC))
        ks.QS=[numpy.einsum('ik,kj->ij',Q2,S),numpy.einsum('ik,kj->ij',Q3,S)]
        ks.SQ=[numpy.einsum('ik,kj->ij',S,Q2),numpy.einsum('ik,kj->ij',S,Q3)]

      # DEBUG TEST 
      test = numpy.dot(Q2,numpy.dot(S,Q2))-Q2
      test+= numpy.dot(Q3,numpy.dot(S,Q3))-Q3
      test+= numpy.dot(Q2,numpy.dot(S,Q3))
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q2),numpy.einsum('ij,ji->',S,Q3))


def old_build_mbproj(ks):
    ''' Build the projection operators QS and SQ from existing basis core AOs 
    Args:
        ks : an instance of :class:`RKS`
    '''  
    if(ks.QS is None):
      m = ks.mol
      S = ks.get_ovlp() 
      N=(S.shape)[0]
      Q    =numpy.zeros((N,N))
      ks.QS=numpy.zeros((N,N))
      ks.SQ=numpy.zeros((N,N))
      # Prepare dummy molecule with minimal basis set 
      md = m.copy()
      md.basis='uncccpcvtz'   
      md.build() 
      SM = md.intor_symmetric('int1e_ovlp')
      
      # Build cross-overlap matrix between current and minimal core 
      SX = gto.intor_cross('int1e_ovlp',m,md)
      #print('Cross- overlap \n',SX.shape)
      
      # Find core minimal basis AOs 
      coreaos=[]
      labs = md.ao_labels()
      for iao in range(md.nao):
        icen = int(labs[iao].split()[0])
        iat = md.atom_charge(icen)
        if(iat>2 and   (' 1s' in labs[iao]) ):
           coreaos.append(iao)
           #print(labs[iao])

      # Build core minimal basis 
      NC = len(coreaos)
      #print(' Keeping ',NC,' core minimal AOs') 
      SC = numpy.zeros((NC,NC))
      SXC = numpy.zeros((N,NC))
      #print('Transfer Shape: ',SXC.shape)
      for ic in range(NC):
        SXC[:,ic] = SX[:,coreaos[ic]]
        for jc in range(NC):
           SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]
      #print('SC ',SC)
      #print('SXC ',SXC)
      SXCT = numpy.transpose(SXC)
      
      # Build and orthogonalize projected core minimal basis set 
      SPC = numpy.dot(SXCT,numpy.dot(S,SXC))
      #print('Projected core min basis: ',SPC.shape)
      #print('SPC ',SPC)
      (vals,vecs) = linalg.eigh(SPC)
      SPCmhalf = numpy.zeros((NC,NC))
      for i in range(NC):
        if(vals[i]>0.00000001):
          SPCmhalf[i,i] = ((vals[i]).real)**(-0.5)
        else:
           print('Eliminating overlap eigenvalue ',i,vals[i])
      QPC = numpy.dot(vecs,numpy.dot(SPCmhalf,numpy.transpose(vecs)))
      
      # Build projector 
      Q0 = numpy.zeros((NC,NC))
      for i in range(NC):
        v = QPC[i]
        Q0 = Q0 + numpy.outer(v,v)
      Q = numpy.dot(SXC,numpy.dot(Q0,SXCT))
      ks.QS=numpy.einsum('ik,kj->ij',Q,S)
      ks.SQ=numpy.einsum('ik,kj->ij',S,Q)

      # DEBUG TEST 
      test = numpy.dot(Q,numpy.dot(S,Q))-Q
      #print('Q \n',Q)
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))

def build_proj(ks):
    ''' Build the projection operators QS and SQ from list of integers or control string paos 
    Args:
        ks : an instance of :class:`RKS`
    '''
    if(ks.paos is None):
      return 
    if(ks.QS is None):
      if(isinstance(ks.paos,str)):
        if('NewCoreAOs' in ks.paos):
          build_mbproj(ks)
          return 
        if('NewDAOs' in ks.paos):
          build_mbproj(ks,daos=True)
          return 
        if('NewFAOs' in ks.paos):
          build_mbproj(ks,faos=True)
          return 
        if('NewVAOs' in ks.paos):
          build_mbproj_2(ks)
          return 
        if('FragAOs' in ks.paos):
          build_mbproj_fragment(ks)
          return 
        if('SpinAOs' in ks.paos):
          build_mbproj_spin(ks)
          return
        if('NewDZVAOs' in ks.paos):
          build_mbproj_3(ks)
          return 
        if('Dum' in ks.paos):
          build_mbproj(ks,dum=True)
          return 
        aoss = [] 
        if('CoreAOs' in ks.paos):
          aoss = assign_cores(ks.mol)
        elif('DAOs' in ks.paos):
          aoss = get_d(ks.mol)
        elif('AllAOs' in ks.paos):
          aoss = [] 
          aoss2 = []
          for i in range(ks.mol.nao):
            aoss2.append(i) 
          aoss.append(aoss2)
      #elif(isinstance(ks.paos,list) and len(ks.paos[0].shape)==2 ):
      elif(isinstance(ks.paos,list) ): 
        pao_proj(ks) # New function May 2024, list of shells of AOs 
        return 
      elif(isinstance(ks.paos,list)):
        aoss = ks.paos 
      else:
        if(len(ks.paos[0].shape) == 2):
          mo_proj(ks)
          return 
        else:
          raise Exception('Not sure what paos is')

      # If we are still here, aoss contains a list of lists of AOs
      # to project onto. Orthogonalize and project. 
      print('AOs to project ',aoss)
      S = ks.get_ovlp() 
      N=(S.shape)[0]
      ks.QS=[]
      ks.SQ=[]

      # Do a symmetric orthogonalization, then assign orthogonalized vectors
      # to sets based on maximum overlap. We'll *assume* that the ith
      # orthogonalized AO equals the ith AO. 
      (vals,vecs) = linalg.eigh(S)
      Smhalf = numpy.zeros((N,N))
      for i in range(N):
        if(vals[i]>0.00000001):
          Smhalf[i,i] = ((vals[i]).real)**(-0.5)
        else:
           print('Eliminating overlap eigenvalue ',i,vals[i])
      Qbig = numpy.dot(vecs,numpy.dot(Smhalf,numpy.transpose(vecs)))
      for iset in range(len(aoss)):
        Qset = numpy.zeros((N,N))
        for i in aoss[iset]:
          v = Qbig[i]
          Qset = Qset + numpy.outer(v,v)
        ks.QS.append(numpy.einsum('ik,kj->ij',Qset,S))
        ks.SQ.append(numpy.einsum('ik,kj->ij',S,Qset))
        test = numpy.dot(Qset,numpy.dot(S,Qset))-Qset
        print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Qset))

######  May 2024 new functions for pDFT+UCI in projected atomic orbitals 
def makeOPAOs(SAO,pAOs,VeepAOs=None):
  opAOs=[]
  SopAOs=[]
  SAOopAOs=[]
  VeeopAOs=[]
  for ishell in range(len(pAOs)):
     pAO = pAOs[ishell]
     #print('Shell ',ishell,' pAOs \n',pAO)
     nproj=pAO.shape[1]
     #print('Shell ',ishell,' nproj ',nproj)
     opAO = numpy.zeros_like(pAO)
     #Sshell = numpy.einsum('mp,mn,nq->pq',pAO,SAO,pAO)
     Sshell = numpy.dot(numpy.transpose(pAO),numpy.dot(SAO,pAO))
     #print('Shell ',ishell,' pAO overlap \n',Sshell)
     (vals,vecs)=numpy.linalg.eigh(Sshell)
     print('PAO shell ',ishell,' overlap eigenvalues \n',vals)

     # Do sinular value decomposition inverse 
     for i in range(len(vals)):
      if(vals[i]>0.000001):
        vals[i]=vals[i]**(-0.5)
      else:
        vals[i]=0
      vecs[:,i]=vecs[:,i]*vals[i]
     opAO = numpy.dot(pAO,vecs)
     #opAO = numpy.einsum('mi,ij,j->mj',pAO,vecs,vals)
     #print('Shell ',ishell,' opAOs \n',opAO)
     opAOs.append(opAO) 
     #SopAOs.append(numpy.einsum('mi,mn,nj->ij',opAO,SAO,opAO))
     SopAOs.append(numpy.dot(numpy.transpose(opAO),numpy.dot(SAO,opAO)))
     #SAOopAOs.append(numpy.einsum('mi,mn->ni',opAO,SAO))
     SAOopAOs.append(numpy.dot(SAO,opAO))

     # Transform VeepAOs
     if(VeepAOs is not None):
       VeepAO = VeepAOs[ishell]
       temp1=numpy.einsum('pqrs,sl->pqrl',VeepAO,vecs)
       temp2=numpy.einsum('pqrl,rk->pqkl',temp1,vecs)
       temp1=numpy.einsum('pqkl,qj->pjkl',temp2,vecs)
       VeeopAO=numpy.einsum('pjkl,pi->ijkl',temp1,vecs)
       VeeopAOs.append(VeeopAO)
  return(opAOs,SopAOs,SAOopAOs,VeeopAOs)

def makeallOPAOs(SAO,opAOs):
  # Return the rectangular (ao,opao) matrix of all the block-orthogonalized
  # opAOs expressed in the AO basis, the square matrix of opAO-opAO overlaps,
  # and the rectangular matrix of AO-opAO overlaps. 
  nshell = len(opAOs)
  nao = opAOs[0].shape[0]
  ntot=0
  for ishell in range(nshell):
    ntot = ntot + opAOs[ishell].shape[1]
  allopAO=numpy.zeros((nao,ntot))
  j=0
  for ishell in range(nshell):
   for iao in range(opAOs[ishell].shape[1]):
     allopAO[:,j]=(opAOs[ishell])[:,iao]
     j=j+1
  #SAOallopAO = numpy.einsum('mi,mn->ni',allopAO,SAO)
  SAOallopAO = numpy.dot(SAO,allopAO)
  #SallopAO = numpy.einsum('mi,mn,nj->ij',allopAO,SAO,allopAO)
  SallopAO = numpy.dot(numpy.transpose(allopAO),numpy.dot(SAO,allopAO))
  return(allopAO,SallopAO,SAOallopAO)

def pao_proj(ks,pAOs=None,doret=False):
    ''' 
    Build a projection operator Q from an input list of blocks of nonorthogonal
    functions expanded in the current AO basis set
    Args:
        ks : an instance of :class:`RKS`
    '''  
    if(pAOs is None):
      pAOs = ks.paos 
    print('Now in pao_proj with ',len(pAOs),' shells of pAOs ')
    m = ks.mol
    nao = m.nao
    S = ks.get_ovlp() 
    opAOs,SopAOs,SAOopAOs,VeeopAOs = makeOPAOs(S,pAOs)
    opAO,SopAO,SAOopAO = makeallOPAOs(S,opAOs)
    #SopAOm = numpy.linalg.inv(SopAO)
    SopAOm = numpy.linalg.pinv(SopAO)
    Sm = numpy.linalg.inv(S)
    #Q = numpy.einsum('mn,ni,ij,oj,op->mp',Sm,SAOopAO,SopAOm,SAOopAO,Sm)
    temp=numpy.dot(Sm,SAOopAO)
    Q = numpy.dot(numpy.dot(temp,SopAOm),numpy.transpose(temp))
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('PAO_PROJ TEST: ',numpy.sum(test*test))
    if(not doret):
      ks.QS=[numpy.dot(Q,S)]
      ks.SQ=[numpy.dot(S,Q)]
    else:
      return(Q)

def V_1to2(V1,S12,Sm2):
  # General function to convert vector v from basis 1 to basis 2 
  # given their overlap S12 and basis 2 inverse 
  temp=numpy.dot(S12,Sm2)
  print('TEST ',S12.shape,Sm2.shape,temp.shape)
  ret = numpy.dot(V1,temp)
  return(ret)

def P_1to2(P1,S12,Sm2):
  # General function to convert density matrix P from basis 1 to basis 2 
  # given their overlap S12 and basis 2 inverse 
  #P2 = numpy.einsum('pr,mr,smn,nt,tq->spq',Sm2,S12,P1,S12,Sm2)
  temp=numpy.dot(S12,Sm2)
  P2a = numpy.dot(numpy.transpose(temp),numpy.dot(P1[0],temp))
  P2b = numpy.dot(numpy.transpose(temp),numpy.dot(P1[1],temp))
  P2 = numpy.asarray([P2a,P2b])
  return(P2)

def O1_1to2(O1,S12,Sm1):
  # General function to convert one-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  #O2 = numpy.einsum('mi,mn,no,op,pj->ij',S12,Sm1,O1,Sm1,S12)
  temp=numpy.dot(Sm1,S12)
  O2 = numpy.dot(numpy.transpose(temp),numpy.dot(O1,temp))
  return(O2)

def O2_1to2(O1,S12,Sm1):
  # General function to convert two-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  tt = numpy.einsum('mi,mn->ni',S12,Sm1)
  tmp =numpy.einsum('mi,nj,mnop->ijop',tt,tt,O1) 
  O2 =numpy.einsum('ok,pl,ijop->ijkl',tt,tt,tmp) 
  return(O2)


def eci(ks,QS=None,QMOs=None,Pin=None,hl=0):
  # Generate the projected CI correlation energy with Vee projection Q,
  # defaulting to that in ks. October 2025, accept four projections QMO to
  # choose the occa,virta,occb,virtb MOs entering the CI. November 2025, try to
  # add projected MP2 correlation, by projecting the regular MOs (not the
  # occa,virta,occb,virtb MOs of the CI) onto Q. 
  print('Now in ECI')
  if(QS is None):
    QS = ks.QS[0]
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  Q = numpy.dot(QS,Sm) 
  if(QMOs is None):
    QMOs = [Q,Q,Q,Q]
  SQ = numpy.dot(S,Q)
  QSQ = numpy.dot(QS,Q)
  print('eci qsqtest ',numpy.einsum('ij->',QSQ-Q))
  SQSMOs = []
  for ttype in range(4):
    SQSMO = numpy.dot(S,numpy.dot(QMOs[ttype],S))
    SQSMOs.append(SQSMO)

  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # Convert ROHF to UHF 
      mo_a=ks.mo_coeff
      mo_b=mo_a
      e_a = ks.mo_energy 
      e_b=e_a
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec

  # Define function to build reference system's effective onelectron Hamiltonian and projected energy 
  def getReferenceSystem(Pin):
    h0 =  ks.get_hcore()
    Enuc = ks.energy_nuc()
    Pp = numpy.einsum('ij,sjk,kl->sil',QS,Pin,SQ)
    print('Getting reference potential \nTotal and projected electrons: ',numpy.einsum('sij,ij->s',Pin,S),numpy.einsum('sij,ij->s',Pp,S))
    j = ks.get_j(dm=P)
    J = j[0]+j[1]
    j = ks.get_j(dm=Pp)
    Jp0 = j[0]+j[1]
    K  = -1.0*ks.get_k(dm=P)
    Kp0 = -1.0*ks.get_k(dm=Pp)
    n, EXC, VXC = ks._numint.nr_uks(m,ks.grids,ks.xc,P)
    pxc = ks.xc 
    if(ks.allc>0):
      pxc=(pxc.split(','))[0] + ','
    np,EXCp,VXCp0= ks._numint.nr_uks(m,ks.grids,pxc,Pp)
 
    # Project the potentials as well, so full 4-index (ab|cd) is projected 
    Jp= numpy.einsum('ik,kj->ij',SQ,numpy.einsum('ik,kj->ij',Jp0,QS))
    Kp= numpy.einsum('ik,skj->sij',SQ,numpy.einsum('sik,kj->sij',Kp0,QS))
    VXCp= numpy.einsum('ik,skj->sij',SQ,numpy.einsum('sik,kj->sij',VXCp0,QS))

    # Terms 'inside' and 'outside' the projected region 
    Jout   = J-Jp 
    VXCout = (VXC-VXCp) + hyb*(K-Kp) 

    # Mean-field effective Hamiltonian, 'outside' and 'inside' 
    # NOTE this assumes 100% projected Vee 'inside' reference system 
    h1aoout = numpy.zeros_like(P) 
    h1aoout[0] = h0 + Jout + VXCout[0]
    h1aoout[1] = h0 + Jout + VXCout[1] 
    
    EH = numpy.einsum('sij,ji->',P,0.5*J)
    EHp = numpy.einsum('sij,ji->',Pp,0.5*Jp0)
    EX = numpy.einsum('sij,sji->',P,0.5*K)
    EXp = numpy.einsum('sij,sji->',Pp,0.5*Kp0)
  
    # Reference system mean-field two-electron energy and HXC operator 
    EEref = EHp + EXp 
    print('ECI real and ref Hartree and exchange energy ',EH,EHp,EX,EXp)
    print('ECI real and ref EE energy ',EH+EX,EEref)
    vhxcin  = numpy.zeros_like(P) 
    vhxcin[0] = Jp + Kp[0]
    vhxcin[1] = Jp + Kp[1] 
    print('ECI Self energy EHp, <h1out>, <vhxcin> ',EHp,numpy.einsum('sij,sij->',P,h1aoout),numpy.einsum('sij,sij->',P,vhxcin)) 

    # Projected energy
    Eproj0 = numpy.einsum('sij,ij->',P,h0) + Enuc 
    Eproj1 = Eproj0 + EH + EXp   + (EXC - EXCp) + hyb*(EX-EXp) -EEref 

    # Reference system energy, single-determinant HF calc 
    # Note this is *not* the real system HF energy, it's the term we use to
    # compute the reference system correlation energy. 
    ErefHF = EEref +numpy.einsum('sij,sij->',P,h1aoout)
    print('Reference system (not real system) HF energy ',ErefHF)
  
    return(h1aoout,vhxcin,Eproj1,EEref,ErefHF)
  
  # Get embedding potential from SCF density  
  h1aoout,vhxcin,Eproj1,EEref,ErefHF = getReferenceSystem(P)
  EPDFT1 = Eproj1+EEref # Total energy of the real system from a single-determiant HF calc 
  fao = h1aoout + vhxcin 
  fmoa=numpy.einsum('mi,mn,nj->ij',mo_a,fao[0],mo_a)
  fmob=numpy.einsum('mi,mn,nj->ij',mo_b,fao[1],mo_b)
  
  t1=(fmoa-numpy.diag(e_a))**2 + (fmob-numpy.diag(e_b))**2
  print('MORE ',EPDFT1,ks.e_tot)
  print('CI TESTS: ',(EPDFT1-ks.e_tot)*2,numpy.einsum('ij->',t1)) # Test ref sys energy and Fock operator 

  # Transform the MOs, so that we can only do CI with transformed MOs that have
  # non-negligible projection onto Q (or QMO).  Transform occ alpha, virt alpha, occ
  # beta, virt beta blocks separately
  thresh=0.01
  ncas = 0 
  nelecas = 0 
  froz_a=[]
  act_a=[]
  virt_a=[]
  froz_b=[]
  act_b=[]
  virt_b=[]
  for ttype in range(4):  # alpha occ, alpha virt, beta occ, beta virt 
    mo=mo_a[:,:Na]
    if(ttype == 1): mo = mo_a[:,Na:]
    if(ttype == 2): mo = mo_b[:,:Nb]
    if(ttype == 3): mo = mo_b[:,Nb:]
    if(mo.shape[1]>0):
      B = numpy.einsum('mi,mn,nj->ij',mo,SQSMOs[ttype],mo) 
      val,vec= linalg.eigh(B)
      print(' ttype ',ttype,' eigenvals \n',val)
      tt= numpy.einsum('mi,ij->mj',mo,vec) 
      froz=[]
      act=[]
      # TEST add beta virts to keep act_a and act_b the same length 
      if(ttype==3):
        nact=len(act_a) - len(act_b)
        print('Adding ',nact,' of ',B.shape[0],' beta virts to ',len(act_b))
        nvir= B.shape[0]-nact
        for i in range(B.shape[0]):
           vv = tt[:,i]
           if(i<nvir):
             froz.append(vv)
           else:       
             act.append(vv)
      else:           
        for i in range(B.shape[0]):
          #vv = numpy.einsum('p,mp->m',vec[:,i],mo)
          vv = tt[:,i]
          if(val[i]>thresh):
            act.append(vv)
          else:
            froz.append(vv)

      if(ttype == 0): 
        froz_a = froz 
        act_a = act 
      if(ttype == 1): 
        virt_a = froz 
        act_a = act_a + act 
      if(ttype == 2): 
        froz_b = froz 
        act_b = act 
      if(ttype == 3): 
        virt_b = froz 
        act_b = act_b + act 

  ncas = len(act_a)
  nelecas=[Na-len(froz_a),Nb-len(froz_b)]
  # Active orbital resize
  diff=len(act_b)-len(act_a)
  if(diff>0):
    act_b=act_b[diff:]
  tmo_a = numpy.transpose(numpy.array(froz_a + act_a + virt_a))
  atmo_a = numpy.transpose(numpy.array(act_a ))
  tmo_b = numpy.transpose(numpy.array(froz_b + act_b + virt_b))
  atmo_b = numpy.transpose(numpy.array(act_b ))
  print('act_a len',len(act_a))
  print('atmo_a shape ',atmo_a.shape)

  # Expensive test case. Do full CI with all orbitals, for testing purposes only!
  if(hl>0):
    print('ECI doing full CI ')
    ncas = nao
    nelecas=[Na,Nb]
    tmo_a = mo_a 
    tmo_b = mo_b 
    atmo_a = tmo_a
    atmo_b = tmo_b

  print('Doing CI with hl ',hl,' nao ',nao,' nelec ',Na+Nb,' ncas ',ncas,' nelecas ',nelecas)
  if(nelecas[0]+nelecas[1]<2):
    return(0,0) 
  print('Transformed MO shape ',tmo_a.shape)
  #print('Active transformed MO shape ',atmo_a.shape,atmo_b.shape)
  

  # Construct MO-basis reference system h1 and h2 
  h1mo = numpy.zeros((2,ncas,ncas))
  h1mo[0]=numpy.einsum('mi,mn,nj->ij',atmo_a,h1aoout[0],atmo_a)
  h1mo[1]=numpy.einsum('mi,mn,nj->ij',atmo_b,h1aoout[1],atmo_b)
  pmo_a=numpy.dot(QS,atmo_a) 
  pmo_b=numpy.dot(QS,atmo_b)
  porbs = numpy.array([pmo_a,pmo_b])
  apmo_a=numpy.dot(QS,atmo_a)
  apmo_b=numpy.dot(QS,atmo_b)
  aporbs = numpy.array([apmo_a,apmo_b])

  mc=mcscf.UCASCI(ks,ncas,nelecas)
  max_memory = max(400, mc.max_memory-lib.current_memory()[0])
  #h2mo = mc.get_h2eff(porbs)
  h2mo = mc.get_h2eff(aporbs) # Only need in the active projected MOs , right? 
  print('h1mo shape ',h1mo.shape)
  print('h2mo[0] shape ',h2mo[0].shape)

  # Do full CI 
  ErefCI, fcivec = mc.fcisolver.kernel(h1mo, h2mo, mc.ncas, mc.nelecas,
                                      ci0=None,verbose=10, max_memory=max_memory,ecore=0)
  hfvec=numpy.zeros_like(fcivec)
  hfvec[0,0]=1
  eci = mc.fcisolver.energy(h1mo,h2mo,fcivec,mc.ncas,mc.nelecas)
  ehf = mc.fcisolver.energy(h1mo,h2mo,hfvec,mc.ncas,mc.nelecas)
  Ecorr = eci - ehf 

  # Project regular MOs (not transformed MOs defining the CI space) onto Q and
  # generate projected MP2 correlation from HF state 
  nmo = mo_a.shape[1]
  pMO = numpy.zeros((2,nao,nmo))
  pMO[0] = numpy.dot(QS,mo_a) # (ao,mo) 
  pMO[1] = numpy.dot(QS,mo_b)
  mykmp =DFURDMP2(ks,tau=ks.mp2lam,auxbasis='cc-pvtz-ri')
  #teheris = mykmp.ao2mo(mo_coeff=pMO)
  #EMP2 = mykmp.init_amps(mo_energy=ks.mo_energy, mo_coeff=ks.mo_coeff,eris=teheris)[0]
  mykmp.mo_coeff=pMO
  EMP2 = mykmp.calculate_energy()

  print('ECI CASCI and MP2 Ecorr ',Ecorr,EMP2)
  return(Ecorr,EMP2) 

def get_ehxc(ks,P,xc):
  # This function generates the Hartree-exchange-correlation energy for a
  # global hybrid.
  #
  # Returns Hartree plus one-electron energy Eother, exact
  # exchange, DFT exchange, DFT correlation, and total EXC 
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
  (Na,Nb)=m.nelec
  (Pa,Pb)=P

  h0=ks.get_hcore()
  Jmat2 = ks.get_j(dm=P)
  Jmat = Jmat2[0] + Jmat2[1] 
  Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
  K=ks.get_k(dm=P)
  EX=-0.5*numpy.einsum('sij,sij->',P,K)

  x=xc
  x=(x.split(','))[0] + ','

  EXSL=0
  ECSL=0
  NA=0
  hermi=1 
  tiny = 0.00000001
  ni = ks._numint
  ao_deriv=0
  xctype=ni._xc_type(xc)
  if xctype == 'GGA':
    ao_deriv=1
  elif xctype == 'MGGA':
    ao_deriv=1
  if(not(xc=='HF,' or xc=='hf,')):
    # Do the numerical integrals 
    make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
    for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):

      # Densities needed 
      rho_a = make_rhoa(0, aos, mask, xctype)
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

      # Semilocal exchange and correlation 
      exc = ni.eval_xc_eff(xc, rho, deriv=0, xctype=xctype)[0]
      exc = exc*(rhoa+rhob) 
      ex = numpy.zeros_like(exc)
      if(not(x=='HF,' or x=='hf,')):
        ex = ni.eval_xc_eff(x, rho, deriv=0, xctype=xctype)[0]
        ex = ex*(rhoa+rhob) 
      ec = exc - ex 

      # Sum terms 
      EXSL = EXSL + numpy.dot(ex,weight)
      ECSL = ECSL + numpy.dot(ec,weight)
      NA = NA + numpy.dot(rhoa,weight)
  print('GET_EHXC TEST: ',Na,NA)
  EXC = hyb*EX +EXSL +ECSL
  return(Eother,EX,EXSL,ECSL,EXC)


def epzlh(ks,P,allc=0):
  # Generate the pDFT projected XC energy using local-hybrid-PZ ansatz
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  #P = ks.make_rdm1() 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
  (Na,Nb)=m.nelec
  (Pa,Pb)=P

  PP=numpy.zeros_like(P)
  PP[0] = numpy.dot(ks.QS[0],numpy.dot(P[0],ks.SQ[0]))
  PP[1] = numpy.dot(ks.QS[0],numpy.dot(P[1],ks.SQ[0]))
  h0=ks.get_hcore()
  Jmat2 = ks.get_j(dm=P)
  Jmat = Jmat2[0] + Jmat2[1] 
  Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
  K=ks.get_k(dm=P)
  EX=-0.5*numpy.einsum('sij,sij->',P,K)
  KP=ks.get_k(dm=PP)
  EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
  print('EPZSL N ',numpy.einsum('sij,ji->s',P,S),numpy.einsum('sij,ji->s',PP,S))
  #print('EPZSL EX ',EX,EXP)

  EXC1 = EXP+hyb*(EX-EXP)
  EXC2 = EXC1

  R=numpy.zeros_like(P)
  R[0]=0.5*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
  R[1]=0.5*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))
  RP=numpy.zeros_like(P)
  RP[0]=0.5*(numpy.dot(Sm,numpy.dot(KP[0],PP[0])) + numpy.dot(PP[0],numpy.dot(KP[0],Sm)))
  RP[1]=0.5*(numpy.dot(Sm,numpy.dot(KP[1],PP[1])) + numpy.dot(PP[1],numpy.dot(KP[1],Sm)))

  # Do the numerical integrals 
  EX2=0
  EXP2=0
  ESL=0
  EXSL=0
  ECSL=0
  EPSL=0
  EPSL2=0
  NA=0
  NPA=0

  hermi=1 
  xc=ks.xc
  pxc=xc
  if(allc>0):
   pxc=(pxc.split(','))[0] + ','
  ni = ks._numint
  ao_deriv=0
  xctype=ni._xc_type(xc)
  if xctype == 'GGA':
    ao_deriv=1
  elif xctype == 'MGGA':
    ao_deriv=1
  nao=m.nao
  tiny = 0.00000001
  make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
  make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
  make_rhpa, nset = ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[:2]
  make_rhpb       = ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0]
  for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):


      # SL XC 
      rho_a = make_rhoa(0, aos, mask, xctype)
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
      exc, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
      exc = exc*(rhoa+rhob) 
      exsl = exc
      if(allc>0):
        if(pxc == 'hf,'):
          exsl = exsl * 0.0 
        else:   
          exsl, vxc = ni.eval_xc_eff(pxc, rho, deriv=1, xctype=xctype)[:2]
          exsl = exsl*(rhoa+rhob) 
         
      # Projected SL XC 
      rhp_a = make_rhpa(0, aos, mask, xctype)
      rhp_b = make_rhpb(0, aos, mask, xctype)
      if(len(rhp_b.shape)>0):
        rhp_b[0,rhp_b[0]<tiny]=tiny
        rhpb = rhp_b[0]
      else:
        rhp_b[rhp_b<tiny]=tiny
        rhpb = rhp_b
      if(len(rhp_a.shape)>0):
        rhp_a[0,rhp_a[0]<tiny]=tiny
        rhpa = rhp_a[0]
      else:
        rhp_a[rhp_a<tiny]=tiny
        rhpa = rhp_a
      rhp = (rhp_a, rhp_b)
      if(pxc == 'hf,'):
        epxc = 0.0 * exc
      else: 
        epxc, vxc = ni.eval_xc_eff(pxc, rhp, deriv=1, xctype=xctype)[:2]
        epxc = epxc*(rhpa+rhpb)

      # Local hybrid type weighting 
      aos = m.eval_gto("GTOval_sph",coords)
      rhoa= numpy.einsum('ri,ij,rj->r',aos,Pa,aos)
      rhob= numpy.einsum('ri,ij,rj->r',aos,Pb,aos)
      ex=numpy.einsum('ri,sij,rj->r',aos,R,aos)
      ex[ex<tiny]=tiny
      expr=numpy.einsum('ri,sij,rj->r',aos,RP,aos)
      wt = expr/ex
      #wt[wt>1]=1
      #wt[wt<0]=0
      epxc2 = exsl*wt

      EX2 = EX2 -0.5*numpy.dot(ex,weight)
      EXP2 = EXP2 -0.5*numpy.dot(expr,weight)
      ESL = ESL + numpy.dot(exc,weight)
      EXSL = EXSL + numpy.dot(exsl,weight)
      ECSL = ECSL + numpy.dot(exc-exsl,weight)
      EPSL = EPSL + numpy.dot(epxc,weight)
      EPSL2 = EPSL2 + numpy.dot(epxc2,weight)
      NA = NA + numpy.dot(rhoa,weight)
      NPA= NPA+ numpy.dot(rhoa*wt,weight)
  print('EPZLH TEST: ',Eother,EX,EX2,EXP,EXP2,NA,NPA)
  print('EPZLH ESL: ',ESL,EXSL,ECSL,EPSL,EPSL2)
  EXC1 = EXC1+(ESL-EPSL2)
  EXC2 = EXC2+(ESL-EPSL)
  #return(Eother,EXC1,EXC2)
  return(Eother,EX,EXP,EXSL,ECSL,EPSL2)



def new_epzlh(ks,P,lhexp=20):
  # Generate the pDFT projected XC energy using local-hybrid-PZ ansatz
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  #P = ks.make_rdm1() 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
  (Na,Nb)=m.nelec
  (Pa,Pb)=P

  PP=numpy.zeros_like(P)
  PP[0] = numpy.dot(ks.QS[0],numpy.dot(P[0],ks.SQ[0]))
  PP[1] = numpy.dot(ks.QS[0],numpy.dot(P[1],ks.SQ[0]))
  h0=ks.get_hcore()
  Jmat2 = ks.get_j(dm=P)
  Jmat = Jmat2[0] + Jmat2[1] 
  Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
  K=ks.get_k(dm=P)
  EX=-0.5*numpy.einsum('sij,sij->',P,K)
  KP=ks.get_k(dm=PP)
  EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
  print('EPZSL N ',numpy.einsum('sij,ji->s',P,S),numpy.einsum('sij,ji->s',PP,S))
  print('EPZSL EX ',EX,EXP)

  EXC1 = EXP+hyb*(EX-EXP)
  EXC2 = EXC1

  R=numpy.zeros_like(P)
  R[0]=0.5*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
  R[1]=0.5*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))
  RP=numpy.zeros_like(P)
  RP[0]=0.5*(numpy.dot(Sm,numpy.dot(KP[0],PP[0])) + numpy.dot(PP[0],numpy.dot(KP[0],Sm)))
  RP[1]=0.5*(numpy.dot(Sm,numpy.dot(KP[1],PP[1])) + numpy.dot(PP[1],numpy.dot(KP[1],Sm)))

  # Do the numerical integrals 
  EX2=0
  EXP2=0
  EXSL=0
  EXLH=0
  ECSL=0
  ECLH=0
  EXPSL=0
  ECPSL=0
  EXPLH=0
  ECPLH=0
  NA=0
  NPA=0

  hermi=1 
  xc=ks.xc
  pxc = xc 
  pxc=(pxc.split(','))[0] + ','

  ni = ks._numint
  ao_deriv=0
  xctype=ni._xc_type(xc)
  if xctype == 'GGA':
    ao_deriv=1
  elif xctype == 'MGGA':
    ao_deriv=1
  nao=m.nao
  tiny = 0.00000001
  make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
  make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
  make_rhpa, nset = ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[:2]
  make_rhpb       = ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0]
  for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):


      # Density and projected density 
      rho_a = make_rhoa(0, aos, mask, xctype)
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
      rhp_a = make_rhpa(0, aos, mask, xctype)
      rhp_b = make_rhpb(0, aos, mask, xctype)
      if(len(rhp_b.shape)>0):
        rhp_b[0,rhp_b[0]<tiny]=tiny
        rhpb = rhp_b[0]
      else:
        rhp_b[rhp_b<tiny]=tiny
        rhpb = rhp_b
      if(len(rhp_a.shape)>0):
        rhp_a[0,rhp_a[0]<tiny]=tiny
        rhpa = rhp_a[0]
      else:
        rhp_a[rhp_a<tiny]=tiny
        rhpa = rhp_a
      rhp = (rhp_a, rhp_b)

      # Exact exchange 
      aos = m.eval_gto("GTOval_sph",coords)
      #aos = m.eval_gto("GTOval_cart",coords)
      print('TEST shapes ',m.nao,aos.shape,Pa.shape)
      rhoa= numpy.einsum('ri,ij,rj->r',aos,Pa,aos)
      rhob= numpy.einsum('ri,ij,rj->r',aos,Pb,aos)
      rhoa[rhoa<tiny]=tiny
      rhob[rhob<tiny]=tiny
      ex=numpy.einsum('ri,sij,rj->r',aos,R,aos)
      ex[ex<tiny]=tiny
      ex= -0.5*ex
      expr=numpy.einsum('ri,sij,rj->r',aos,RP,aos)
      expr[expr<tiny]=tiny
      expr= -0.5*expr

      # SL XC  
      exsl=numpy.zeros_like(rhoa)
      ecsl=numpy.zeros_like(rhoa)
      if(xc!='hf'):
        excsl, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
        excsl = excsl*(rhoa+rhob) 
        if(pxc == 'hf,'):
          exsl = numpy.zeros_like(excsl)
        else:
          exsl, vxc = ni.eval_xc_eff(pxc, rho, deriv=1, xctype=xctype)[:2]
        exsl = exsl*(rhoa+rhob) 
        ecsl = excsl - exsl 

      # No XC where projected exact exchange is near total exact exchange. 
      fac = expr/ex
      # March 2026 this could be physically >1 where |<Veeproj>| > |<Veefull>|  fac[fac>1]=1
      fac[fac<0]=0

      # Local hybrid weights: No SL XC where fac is near 0
      # Note that exsl must be multiplied by (1-hyb) for global hybrids. 
      ahflh=numpy.zeros_like(ex)
      if(pxc != 'hf,'):
        exslf = exsl
        if(hyb>0.000001):
          exslf = exsl/(1-hyb)
        z=(exslf/ex)-1
        z[z<tiny]=tiny
        ahflh=erf(lhexp*z) 
        # October 24. Local hybrid admixture of full HF exchange near projected regions where fac is 1
        ahflh = fac**0.1
        #ahflh = ahflh*0.0 # turh off 
        ahflh[ahflh>1]=1
        ahflh[ahflh<0]=0
      # Projected SL XC 
      #epxcsl, vxc = ni.eval_xc_eff(xc, rhp, deriv=1, xctype=xctype)[:2]
      #epxcsl = epxcsl*(rhpa+rhpb)
      #epxlh, vxc = ni.eval_xc_eff(pxc, rhp, deriv=1, xctype=xctype)[:2]
      #epxlh = epxlh*(rhpa+rhpb)
      #epcsl = epxcsl - epxlh 
      exlh = ahflh*ex+(1-ahflh)*(hyb*ex+exsl) # exsl is already scaled by (1-hyb), this is a local hybrid of a global hybrid 
      eclh = (1-ahflh)*ecsl
      epxlh = fac*exlh
      epclh = fac*eclh
      epxsl = fac*exsl
      epcsl = fac*ecsl

      EX2   = EX2  + numpy.dot(ex,weight)
      EXP2  = EXP2 + numpy.dot(expr,weight)
      EXSL  = EXSL + numpy.dot(exsl,weight) 
      ECSL  = ECSL + numpy.dot(ecsl,weight)
      EXPSL = EXPSL + numpy.dot(epxsl,weight) 
      ECPSL = ECPSL + numpy.dot(epcsl,weight) 
      EXLH  = EXLH + numpy.dot(exlh,weight) 
      EXPLH = EXPLH + numpy.dot(epxlh,weight) 
      ECLH  = ECLH + numpy.dot(eclh,weight)
      ECPLH = ECPLH + numpy.dot(epclh,weight) 
      NA = NA + numpy.dot(rhoa+rhob,weight)
      NPA= NPA+ numpy.dot((rhoa+rhob)*fac,weight)
  # For global hybrids the semilocal exchange is already scaled by (1-hyb)
  if(hyb is not None):
     if(hyb>0.000001 and hyb<1):
       EXSL = EXSL/(1-hyb) 
       EXPSL = EXPSL/(1-hyb) 
       EXPLH = EXPLH/(1-hyb) 
  print('EPZLH Ntot,Nact',NA,NPA)
  print('EPZLH TEST: ',Eother,EX,EX2,EXP,EXP2,NA,NPA)
  print('EPZSL ESL: ',EXSL,ECSL,EXPSL,ECPSL)
  print('EPZLH ELH: ',EXLH,ECLH,EXPLH,ECPLH)
  #return(Eother,EX,EXP, EXSL,EXLH,ECSL,ECLH, EXPLH,ECPLH)
  #return(Eother,EX,EXP, EXLH,ECSL,EXPLH,ECPSL)
  return(Eother,EX,EXP, EXSL,EXPSL,ECSL,ECPSL)

def puks(ks,xc,S,Pin,wts,Js,Qs,SQs,QSs):
  # Generate semilocal XC energy and potential from Pin, weighted by projected
  # density matrices 
  lam = ks.lhlam
  if(lam is None):
    lam=1.0 
  m = ks.mol
  cc = 1.51525 
  nproj=len(wts)
  dor35=False 

  # R35 projections for projected semilocal XC energy density
  ar35s=[100,50,30,20,10,5,2,1,0.3,0.1,0.05,0.01,0.005]
  nr35=len(ar35s) 
  Sr35 = None 
  Smr35 = None 
  if(dor35): 
    Sr35= numpy.zeros((nr35,nr35))
    mss=[]
    for amu in(ar35s):
      mss.append(stretchAOs(m,amu))
    for mu in range(nr35):
      amu = ar35s[mu]
      for nu in range(nr35):
        anu = ar35s[nu]
        Sr35[mu,nu] = (2*(amu*anu)**0.5/(amu+anu))**1.5 
    Smr35 = numpy.linalg.inv(Sr35)

  nao = m.nao
  ni = ks._numint
  xctype=ni._xc_type(xc)
  cutoff = ks.grids.cutoff * 1e2
  nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(ks.grids.cutoff))
  ao_loc = m.ao_loc_nr()
  pair_mask = m.get_overlap_cond() < -numpy.log(ni.cutoff)
  ao_deriv=0
  if xctype == 'GGA':
    ao_deriv=1
  elif xctype == 'MGGA':
    ao_deriv=1
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  ES0=0
  ES =0
  VSRHO0 =numpy.zeros_like(Pin)
  VSRHO  =numpy.zeros_like(Pin)
  VSRHOP =numpy.zeros_like(Pin)
  VSG    =numpy.zeros_like(Pin) 
  tiny = 0.00000001
  if(xc=='hf,' or xc=='HF,'):
    return(ES,VSRHO+VSRHOP+VSG)

  # Density makers 
  #print('Input density dimensions ',Pin.shape)
  hermi=1 
  make_rhoa, nset = ni._gen_rho_evaluator(m, [Pin[0]], hermi, False, ks.grids)[:2]
  make_rhob       = ni._gen_rho_evaluator(m, [Pin[1]], hermi, False, ks.grids)[0]
  #print('The number of sets is ',nset)
  make_rhpas=[]
  make_rhpbs=[]
  n=[0,0]
  nps=[]
  ldax=0
  ldaxps=[]
  for itot in range(len(wts)): 
    nps.append([0,0])
    ldaxps.append(0)
    PP = numpy.zeros_like(Pin)
    PP[0] = numpy.dot(QSs[itot],numpy.dot(Pin[0],SQs[itot]))
    PP[1] = numpy.dot(QSs[itot],numpy.dot(Pin[1],SQs[itot]))
    print('PUKS test ',itot,numpy.einsum('sij,ij->s',PP,S),numpy.einsum('sij,ij->s',Pin,S))
    make_rhpas.append(ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[0])
    make_rhpbs.append(ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0])

  # Numerical integration 
  max_memory = ks.max_memory - lib.current_memory()[0]
  aow = None
  for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=max_memory):
    rhoa = make_rhoa(0, aos, mask, xctype)
    rhob = make_rhob(0, aos, mask, xctype)
    ra = rhoa[0]
    rb = rhob[0]
    ao = aos[0]
    if xctype == 'LDA':
      ra=rhoa
      rb=rhob
      ao = aos
    rho = (rhoa, rhob)
    ngrids = coords.shape[0]
    for icoord in range(ngrids):
      if(ra[icoord]<1e-12):
        ra[icoord]=1e-12
      if(rb[icoord]<1e-12):
        rb[icoord]=1e-12
                
    exc, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
    exc = exc*(ra+rb) 
    vxcrho = vxc[0]
    if xctype == 'LDA':
      vxcrho = vxc
    ES0 = ES0 + numpy.dot(exc,weight)
    n[0]+=numpy.dot(weight,ra)
    n[1]+=numpy.dot(weight,rb)
    _dot_ao_ao_sparse(ao, ao, weight*vxcrho[0], nbins, mask, pair_mask, ao_loc,
                      hermi, VSRHO0[0])
    _dot_ao_ao_sparse(ao, ao, weight*vxcrho[1], nbins, mask, pair_mask, ao_loc,
                      hermi, VSRHO0[1])


    # R35 projections, each projected orbital (expanded in AOs) projected onto
    # each R35 grid point function. 
    if(dor35): 
      aor35s=numpy.zeros((nr35,ngrids,nao))
      for ir35 in range(len(mss)):
        molr35 = mss[ir35]
        b = ar35s[ir35]
        aor35 =  ni.eval_ao(molr35, coords, deriv=0, non0tab=mask,
                            cutoff=ks.grids.cutoff)
        aor35s[ir35] = aor35
      Saor35s = numpy.einsum('rs,sgm->rgm',Smr35,aor35s)
      aexps=cc*ra**(0.66667)
      Esls = ((2/math.pi)**0.5)*(aexps**0.5)
      ldafacs=numpy.zeros((ngrids,nr35))
      for mu in range(nr35):
        amu = ar35s[mu]
        ldafacs[:,mu]= (2*(amu*aexps)**0.5/(amu+aexps))**1.5 
      aoldas = numpy.einsum('gr,rgm->gm',ldafacs,Saor35s)
      for itot in range(nproj):
        J = Js[itot] 
        Q = Qs[itot]
        temp = numpy.einsum('gm,mn->gn',aoldas,Q)
        v= numpy.einsum('gm,gm->g',temp,aoldas)
        #ldawts = (0.5*v*v*J/Esls) * wts[itot]
        ldawts = (v)**0.5 * wts[itot]
        ES = ES  + numpy.dot(ldawts*exc,weight)
        nps[itot][0]+=numpy.dot(weight,ra*ldawts)
        nps[itot][1]+=numpy.dot(weight,rb*ldawts)
        for icoord in range(ngrids):
          print('FACS %2d  %12.6f %12.6f %12.6f %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e'%(itot,coords[icoord,0],coords[icoord,1],coords[icoord,2],ra[icoord],v[icoord],ldawts[icoord],-0.5*v[icoord]**2*J,Esls[icoord],exc[icoord],ldawts[icoord]*exc[icoord]))
      
    # Weighted density projections 
    else:
      aow = None 
      for ip in range(len(wts)):
        VSRHOP2 =numpy.zeros_like(Pin)
        make_rhpa = make_rhpas[ip]
        make_rhpb = make_rhpbs[ip]
        rpa = make_rhpa(0, ao, mask, xctype='LDA')
        rpb = make_rhpb(0, ao, mask, xctype='LDA')
        for icoord in range(ngrids):
          if(rpa[icoord]<1e-12):
            rpa[icoord]=1e-12
          if(rpb[icoord]<1e-12):
            rpb[icoord]=1e-12
        num = rpa+rpb+0.0000001
        den = ra+rb+0.0001
        f = wts[ip]*((rpa+rpb)/den)**lam 
        fra = -lam*f/den 
        frb = -lam*f/den 
        frpa = lam*f/num 
        frpb = lam*f/num 
        
        nps[ip][0]+=numpy.dot(weight,rpa)
        nps[ip][1]+=numpy.dot(weight,rpb)
        ES = ES  + numpy.dot(f*exc,weight)
        vsrhop = (frpa*exc,frpb*exc)
        vsrho = vxc 
        vsg = None 
        if xctype == 'GGA':
          vsrho = vxc[:,0,:]
          vsg = vxc[:,1:4,:]
        #for icoord in range(ngrids):
      #  print('FACS %12.6f %12.6f %12.6f   %7.3e %7.3e %7.3e %7.3e   %7.3e %7.3e %7.3e %7.3e %7.3e  %7.3e '%(coords[icoord,0],coords[icoord,1],coords[icoord,2],ra[icoord],rb[icoord],rpa[icoord],rpb[icoord],f[icoord],fra[icoord],frb[icoord],frpa[icoord],frpb[icoord], exc[icoord]))
  
        _dot_ao_ao_sparse(ao, ao, weight*(f*vsrho[0]+fra*exc), nbins, mask, pair_mask, ao_loc,
                            hermi, VSRHO[0])
        _dot_ao_ao_sparse(ao, ao, weight*(f*vsrho[1]+frb*exc), nbins, mask, pair_mask, ao_loc,
                            hermi, VSRHO[1])
        _dot_ao_ao_sparse(ao, ao, weight*vsrhop[0], nbins, mask, pair_mask, ao_loc,
                            hermi, VSRHOP2[0])
        _dot_ao_ao_sparse(ao, ao, weight*vsrhop[1], nbins, mask, pair_mask, ao_loc,
                            hermi, VSRHOP2[1])
        VSRHOP[0] = VSRHOP[0]+numpy.dot(SQs[ip],numpy.dot(VSRHOP2[0],QSs[ip]))
        VSRHOP[1] = VSRHOP[1]+numpy.dot(SQs[ip],numpy.dot(VSRHOP2[1],QSs[ip]))
        if xctype == 'GGA':
          aow = _scale_ao_sparse(aos[1:4], weight*f*vsg[0], mask, ao_loc, out=aow)
          _dot_ao_ao_sparse(aos[0], aow, None, nbins, mask, pair_mask, ao_loc,
                            hermi=0, out=VSG[0])
          aow = _scale_ao_sparse(aos[1:4], weight*f*vsg[1], mask, ao_loc, out=aow)
          _dot_ao_ao_sparse(aos[0], aow, None, nbins, mask, pair_mask, ao_loc,
                            hermi=0, out=VSG[1])
    VSG = lib.hermi_sum(VSG.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nao,nao)

  print('PUKS traces ',n,nps)
  print('PUKS ES0 AND ES',ES0,ES)
  #print('VSRHO0 \n',VSRHO0)
  #print('VSRHO \n',VSRHO)
  #print('VSRHOP \n',VSRHOP)
  return(ES,VSRHO+VSRHOP+VSG)

def euci3(ks,xc,Pin=None,addX=True,addMP2=True,stype=1):
  # Generate the PiFCI correlation energy and potential for each oPAO
  # Pin is the input 1PDM
  # addX=True adds exact exchange and subtracts off semilocal XC for each oPAO 
  # addMP2=True subtracts off MP2 correlation 
  # stype is a switch for projected semilocal XC 
  #  stype=1: Traditional DFT+U, -J/2*(noa + nob) 
  #  stype=2: Pass projected 1PDM to standard DFT integration 
  #  stype=3: Reweight exc[rho] with puks
  #  stype=4: Reweight exc[rho] with puks

  # Set up 
  pAOs = ks.paos 
  m = ks.mol
  (Na,Nb)=m.nelec
  nao = m.nao
  ni = ks._numint
  phyb = ks.phyb
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  print('EUCI3 OMEGA ALPHA HYB ',omega,alpha,hyb)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  N=(S.shape)[0]
  (vals,vecs) = linalg.eigh(S)
  Smhalf = numpy.zeros((N,N))
  for i in range(N):
    if(vals[i]>0.00000001):
      Smhalf[i,i] = ((vals[i]).real)**(-0.5)
      
  # Build UHF-indexed 1PDM, mos, mo energies 
  if(len(P.shape)<3): # RHF to UHF 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # ROHF to UHF 
      raise Exception('euci3 NYI for ROHF')
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec
  print('NUMTEST ',numpy.einsum('sij,ij->s',P,S))

  # Generate the orthogonalized projected AOs with which to build the projected
  # atomic natural orbitals. This part could be done just once, not in every
  # SCF cycle. 
  opAOs,SopAOs,SAOopAOs,VeeopAOs= makeOPAOs(S,pAOs,ks.VeepAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 
  dum1,dum2,dum3,VeeRSopAOs= makeOPAOs(S,pAOs,ks.VeeRSpAOs)

  # Buffers for UHF exchange and correlation potentials 
  VX = numpy.zeros_like(P)
  VC = numpy.zeros_like(P)
  VCP= numpy.zeros_like(P)
  VS = numpy.zeros_like(P)

  # Energy weighted density matrices 
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.dot(mo_a[:,Na:],numpy.transpose(mo_a[:,Na:]))
  Pv[1] = numpy.dot(mo_b[:,Nb:],numpy.transpose(mo_b[:,Nb:]))
  PE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,:Na],e_a[:Na])
  PE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,:Na]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,:Nb],e_b[:Nb])
  PE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,:Nb]))
  PvE=numpy.zeros_like(P)

  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],e_a[Na:])
  PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],e_b[Nb:])
  PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  # Debug test: what if we only use the LUMO? 
  #Pv[0] = numpy.dot(mo_a[:,Na:Na+1],numpy.transpose(mo_a[:,Na:Na+1]))
  #Pv[1] = numpy.dot(mo_b[:,Nb:Nb+1],numpy.transpose(mo_b[:,Nb:Nb+1]))
  #temp=numpy.einsum('mi,i->mi',mo_a[:,Na:Na+1],e_a[Na:Na+1])
  #PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:Na+1]))
  #temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:Nb+1],e_b[Nb:Nb+1])
  #PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:Nb+1]))

  print('Occ and virt traces: ',numpy.einsum('sij,ij->s',P,S),numpy.einsum('sij,ij->s',Pv,S))
  #print('Occ energies:\n',e_a[:Na],e_b[:Nb])
  #print('Virt energies:\n',e_a[Na:],e_b[Nb:])

  # Renormalized virtal DM for renormalized perturbation theory 
  eap = numpy.copy(e_a[Na:])
  ebp = numpy.copy(e_b[Nb:])
  mp2lam = ks.mp2lam
  if(mp2lam is None):
    mp2lam = 2.39 
  for i in range(len(eap)):
    if(eap[i]-e_a[Na-1] <mp2lam):
      eap[i] = e_a[Na-1]+mp2lam
  for i in range(len(ebp)):
    if(ebp[i]-e_b[Nb-1] <mp2lam):
      ebp[i] = e_b[Nb-1]+mp2lam
  PvEp=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],eap)
  PvEp[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],ebp)
  PvEp[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  # Indexing for projected atomic natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  exs=numpy.zeros(ntot)
  ecs=numpy.zeros(ntot)
  ecps=numpy.zeros(ntot)
  ess=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)
  Qs=[] 
  Js=[] 
  SQs=[] 
  QSs=[] 

  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    SopAOm=numpy.linalg.pinv(SopAOs[ishell])
    Vees = VeeopAOs[ishell]
    VeeRSs = VeeRSopAOs[ishell]

    # Project density matrices into this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(Pv,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)
    PvEs=P_1to2(PvE,SAOopAOs[ishell],SopAOm)
    PvEps=P_1to2(PvEp,SAOopAOs[ishell],SopAOm)

    #print('P:\n',P,'\nPs:\n',Ps)

    # Potentials in this shell
    VXs = numpy.zeros_like(Ps)
    VCs = numpy.zeros_like(Ps)
    VCPs= numpy.zeros_like(Ps)
    VSs = numpy.zeros_like(Ps) # Semilocal exchange 

    # Build atomic natural orbitals in this shell 
    (vals,vecs)=numpy.linalg.eigh(Ps[0])
    print('Shell ',ishell,' alpha density eigenvalues ',vals)
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj]
       vv = numpy.outer(v,v)
       v2=numpy.dot(v,SAOopAOs[ishell].T)
       #Q = numpy.outer(v2,v2)
       #Q = numpy.dot(Smhalf,numpy.dot(numpy.outer(v2,v2),Smhalf))
       Q = numpy.dot(Sm,numpy.dot(numpy.outer(v2,v2),Sm))
       #print('Q shape ',Q.shape)
       QS = numpy.dot(Q,S)
       SQ = numpy.dot(S,Q)
       test=numpy.dot(Q,numpy.dot(S,Q))-Q
       print('Q test ',numpy.trace(numpy.dot(test,test)))

       # Weight is its overlap with ALL projected atomic orbitals 
       w0 = numpy.dot(v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = 1.0/numpy.dot(w0,w0)
       wts[itot]=wt
       print('Total Weight',wt)

       # Self-energies 
       J=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,Vees))))
       JLR=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,VeeRSs))))
       JSR = J - JLR 
       Jhyb = (1-hyb)*J
       if(omega>0):
         Jhyb = (1-alpha)*JLR + (1-hyb)*JSR
       if(hyb>0.9999):
         Jhyb = J
       print('Total, long-range, short-range, and hybrid-SL J ',J,JLR,JSR,Jhyb)

       # Occupation numbers and energy-weighted terms
       noa=numpy.dot(v,numpy.dot(Ps[0],v))
       nob=numpy.dot(v,numpy.dot(Ps[1],v))
       nva=numpy.dot(v,numpy.dot(Pvs[0],v))
       nvb=numpy.dot(v,numpy.dot(Pvs[1],v))
       eoa=numpy.dot(v,numpy.dot(PEs[0],v))
       eob=numpy.dot(v,numpy.dot(PEs[1],v))
       eva=numpy.dot(v,numpy.dot(PvEs[0],v))
       evb=numpy.dot(v,numpy.dot(PvEs[1],v))
       evap=numpy.dot(v,numpy.dot(PvEps[0],v))
       evbp=numpy.dot(v,numpy.dot(PvEps[1],v))
       #print('Occupancies ',ishell,iproj,noa,nob,nva,nvb)
       print('Occupancies %2d %2d %7.3f %7.3f '%(ishell,iproj,noa,nob))

       # Projected exact exchange and semilocal XC energy and potential 
       exs[itot]   = -Jhyb*(noa**2+nob**2)/2
       VXs[0] = VXs[0] -Jhyb*noa*vv*wt
       VXs[1] = VXs[1] -Jhyb*nob*vv*wt
       if(addX):
         if(stype==1): # DFT+U-type model, not very good 
           ess[itot]   = -Jhyb*(noa+nob)/2
           VSs[0] = VSs[0] -Jhyb/2*vv*wt
           VSs[1] = VSs[1] -Jhyb/2*vv*wt
         elif(stype>=2):
           Pss0=numpy.asarray([vv*noa,vv*nob])
           Pss=P_1to2(Pss0,SAOopAOs[ishell].T,Sm)
           if(stype==2): # Pass projected 1PDM to XC functional 
             max_memory = ks.max_memory - lib.current_memory()[0]
             np, excp, vxcp0 = ni.nr_uks(m, ks.grids, xc,Pss, max_memory=max_memory)
             ess[itot] = excp
             VS[0]=VS[0] + numpy.dot(SQ,numpy.dot(vxcp0[0],QS))
             VS[1]=VS[1] + numpy.dot(SQ,numpy.dot(vxcp0[1],QS))
             print('De-Projected Test: ',noa,nob,numpy.einsum('sij,ij->s',Pss,S),np)
           elif(stype>=3): # Save Pss to weight the full-P XC functional
             Qs.append(Q)
             Js.append(J)
             SQs.append(SQ)
             QSs.append(QS)
         

       # Projected opposite spin correlation energy and potential 
       if(Na>0 and Nb>0):
         eoa = eoa/(noa +0.00000001) # This un-does a trick we do in computing
# transformed orbitals <psiO|Fock|psiO> , a trick which becomes a problem if nob
# is identically zero because there are no minority-spin electrons. 
         eob = eob/(nob +0.00000001)
         eva = eva/(nva +0.00000001)
         evb = evb/(nvb +0.00000001)
         print('Proj Occ Energies  ',eoa,eob)
         print('Proj Virt Energies ',eva,evb)

         # Diagonalize 
         # Use the more transparent nva=1-noa and express in terms of osq
         # directly. The potential contribution is zero at occupancy 1/2 and
         # pushes towards half-filling. 
         #o = J*(noa*nob*nva*nvb)**0.5 # <Phi_0|Vp(r)|Phi_oo^vv> 
         osq   = J**2*(noa*nob*(1-noa)*(1-nob)) # o = <Phi_0|Vp(r)|Phi_oo^vv> 
         osqoa = J**2*(1-2*noa)*nob*(1-nob)
         osqob = J**2*(1-2*nob)*noa*(1-noa)
         # RETEST 
         osq   = J**2*(noa*nob*nva*nvb) # o = <Phi_0|Vp(r)|Phi_oo^vv> 
         osqoa = J**2*(1-2*noa)*nob*nvb    
         osqob = J**2*(1-2*nob)*noa*nva    
         ee=(eva+evb-eoa-eob)          
         # Don't do this, a bad SCF step might make ee negative ee= numpy.maximum(ee,1e-10)
         d = ee/2 
         doa = 0 
         dob = 0 
         ec = d-(d**2+osq)**0.5 
         ecosq =-0.5*(d**2+osq)**(-0.5)
         ecd = 0 
         eca = ecosq*osqoa + ecd*doa
         ecb = ecosq*osqob + ecd*dob
         ecs[itot]=ec
         VCs[0] = VCs[0] +eca*vv*wt
         VCs[1] = VCs[1] +ecb*vv*wt
         #print('VCs %2d %.3f| %.3f %.3f %.6f %.6f | %.6f %.6f'%(ishell,vv[0,0],noa,nob,VCs[0][0,0],VCs[1][0,0],ecosq,osqoa))

         # Perturbation theory terms
         eep=(evap+evbp-eoa-eob)          
         dp = eep/2 
         ecp = -0.5*osq/dp
         ecposq = -0.5/dp
         ecpa = ecposq*osqoa 
         ecpb = ecposq*osqob 
         ecps[itot]=ecp
         # Don't try to SCF the perturbation theory VCPs[0] = VCPs[0] +ecpa*vv*wt
         #VCPs[1] = VCPs[1] +ecpb*vv*wt
         print('Hamiltonian osq and d and ec',osq,d,ec,dp,ecp)

    # Back-project potentials out of this shell 
    for i in range(2):
      VX[i] = VX[i] + O1_1to2(VXs[i],SAOopAOs[ishell].T,SopAOm)
      VC[i] = VC[i] + O1_1to2(VCs[i],SAOopAOs[ishell].T,SopAOm)
      VCP[i] = VCP[i] + O1_1to2(VCPs[i],SAOopAOs[ishell].T,SopAOm)
      if(stype==1):
        VS[i] = VS[i] + O1_1to2(VSs[i],SAOopAOs[ishell].T,SopAOm)

  print('EXPS  ',exs)
  print('ECPS  ',ecs)
  print('WTS   ',wts)
  EC = numpy.dot(ecs  ,wts)
  ECP= numpy.dot(ecps ,wts)
  EX = numpy.dot(exs  ,wts)
  if(stype==3):
    ES, VS = puks(ks,xc=xc,S=S,Pin=P,wts=wts,Js=Js,Qs=Qs,SQs=SQs,QSs=QSs)
  else:
    print('ESPS  ',ess)
    ES = numpy.dot(ess  ,wts)
  #print('EX AND VX ',EX,'\n',VX)
  #print('EC AND VC ',EC,'\n',VC)
  #print('ES AND VS ',ES,'\n',VS)
  print('EUCI3 EXP ECP ECMP2P ESLP ',EX,EC,ECP,ES)
  EXC = EC
  VXC = VC
  if(addMP2):
   EXC = EXC - ECP 
   VXC = VXC - VCP 
  if(addX):
   EXC = EXC + EX - ES
   VXC = VXC + VX - VS 
  #print('RETURNING VXC ',VXC.shape,'\n',VXC)

  #return(EXC,VXC,EX,EC,ECP,ES)
  return(EXC,VXC,EX,EC,ECP,ES)

def euci2(ks,Pin=None,lam=1):
  # Generate the pDFT+UCI correlation energy 

  # Set up 
  pAOs = ks.paos 
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  PSP=numpy.einsum('sij,jk,skl->sil',P,S,P)
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # ROHF to UHF 
      print('Converting ROHF to UHF') 
      mo_a=ks.mo_coeff
      mo_b=mo_a
      e_a = ks.mo_energy 
      e_b=e_a
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec

  opAOs,SopAOs,SAOopAOs,VeeopAOs= makeOPAOs(S,pAOs,ks.VeepAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 

  K=ks.get_k(dm=P)
  EX=-0.5*numpy.einsum('sij,sij->',K,P)
  R=numpy.zeros_like(P)
  R[0]=0.5*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
  R[1]=0.5*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))

  # Energy weighted density matrices 
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.dot(mo_a[:,Na:],numpy.transpose(mo_a[:,Na:]))
  Pv[1] = numpy.dot(mo_b[:,Nb:],numpy.transpose(mo_b[:,Nb:]))
  PE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,:Na],e_a[:Na])
  PE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,:Na]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,:Nb],e_b[:Nb])
  PE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,:Nb]))
  PvE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],e_a[Na:])
  PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],e_b[Nb:])
  PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  print('Occ and virt traces: ',numpy.einsum('sij,ij->s',P,S),numpy.einsum('sij,ij->s',Pv,S))

  # Indexing for projected natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  exs=numpy.zeros(ntot)
  excsls=numpy.zeros(ntot)
  ecs=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)

  # 1PDM projected onto the full set of states 
  PF=numpy.zeros_like(P)
  PF[0] = numpy.dot(ks.QS[0],numpy.dot(P[0],ks.SQ[0]))
  PF[1] = numpy.dot(ks.QS[0],numpy.dot(P[1],ks.SQ[0]))
  KF=ks.get_k(dm=PF)
  RF=numpy.zeros_like(P)
  RF[0]=0.5*(numpy.dot(Sm,numpy.dot(KF[0],PF[0])) + numpy.dot(PF[0],numpy.dot(KF[0],Sm)))
  RF[1]=0.5*(numpy.dot(Sm,numpy.dot(KF[1],PF[1])) + numpy.dot(PF[1],numpy.dot(KF[1],Sm)))

  # List of projected 1PDMs for numerical integrations 
  PPs=[]
  RPs=[]
  PpSPps=[]

  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    SopAOm=numpy.linalg.pinv(SopAOs[ishell])
    Vees = VeeopAOs[ishell]

    # Density matrices in this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(Pv,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)
    PvEs=P_1to2(PvE,SAOopAOs[ishell],SopAOm)
    print('Shell ',ishell,' alpha density \n',Ps[0])
    print('Shell ',ishell,' projected 1PDM diagonal ')
    for iproj in range(Ps.shape[1]):
      print('%6.3f %6.3f '%(Ps[0,iproj,iproj],Ps[1,iproj,iproj]))
    print('Shell ',ishell,' projected virt1PDM diagonal ')
    for iproj in range(Pvs.shape[1]):
      print('%6.3f %6.3f '%(Pvs[0,iproj,iproj],Pvs[1,iproj,iproj]))

    # Loop over atomic natural orbitals in this shell 
    (vals,vecs)=numpy.linalg.eigh(Ps[0])
    print('Shell ',ishell,' alpha density eigenvalues ',vals)
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj]

       w0 = numpy.dot(v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = numpy.dot(w0,w0)
       wts[itot]=wt
       print('Total Weight Projection ',wt)
       print('Weight ',1/wt)

       J=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,Vees))))

       # Assemble the 2x2 Hamiltonian 
       noa=numpy.dot(v,numpy.dot(Ps[0],v))
       nob=numpy.dot(v,numpy.dot(Ps[1],v))
       nva=numpy.dot(v,numpy.dot(Pvs[0],v))
       nvb=numpy.dot(v,numpy.dot(Pvs[1],v))
       eoa=numpy.dot(v,numpy.dot(PEs[0],v))
       eob=numpy.dot(v,numpy.dot(PEs[1],v))
       eva=numpy.dot(v,numpy.dot(PvEs[0],v))
       evb=numpy.dot(v,numpy.dot(PvEs[1],v))
       print('Occupancies ',ishell,iproj,noa,nob)
       print('Virt occs   ',nva,nvb)
       print('Self energy ',J)

       # Projected exact exchange 
       exs[itot]   = -J*(noa**2+nob**2)/2

       # Projected 1PDMs for projected DFT numerical integration 
       Pn2 = numpy.zeros_like(Ps)
       Pn2[0] = numpy.einsum('i,j->ij',v,v)*noa
       Pn2[1] = numpy.einsum('i,j->ij',v,v)*nob
       Pn1=P_1to2(Pn2,numpy.transpose(SAOopAOs[ishell]),Sm)
       print('Unproj TEST ',noa,nob,numpy.einsum('sij,ij->s',Pn1,S))
       PPs.append(Pn1)
       PpSPp = numpy.einsum('sik,kl,slj->sik',Pn1,S,Pn1)
       print('PpSpP TEST',numpy.einsum('sij,ij->s',PSP,S),numpy.einsum('sij,ij->s',PpSPp,S))
       PpSPps.append(PpSPp)
       Kn1=ks.get_k(dm=Pn1)
       Rp1=numpy.zeros_like(P)
       Rp1[0]=0.5*(numpy.dot(Sm,numpy.dot(Kn1[0],Pn1[0])) + numpy.dot(Pn1[0],numpy.dot(Kn1[0],Sm)))
       Rp1[1]=0.5*(numpy.dot(Sm,numpy.dot(Kn1[1],Pn1[1])) + numpy.dot(Pn1[1],numpy.dot(Kn1[1],Sm)))
       RPs.append(Rp1)

       # Diagonalize the 2x2 Hamiltonian 
       if(noa>0.000001 and nob>0.000001 and nva>0.000001 and nvb>0.000001):
          eoa = eoa/(noa +0.00000001)
          eob = eob/(nob +0.00000001)
          eva = eva/(nva +0.00000001)
          evb = evb/(nvb +0.00000001)
          print('Proj Occ Energies    ',eoa,eob)
          print('Proj Virt Energies',eva,evb)

          # Diagonalize 
          o = J*(noa*nob*nva*nvb)**0.5 # <Phi_0|Vp(r)|Phi_oo^vv> 
          o = numpy.maximum(o,1e-10)
          ee=(eva+evb-eoa-eob)          
          #d = (ee + J*(noa*nob+nva*nvb-noa*nvb-nob*nva))/2
          d = ee/2 # TEST typically little difference 
          ec = d-(d**2+o**2)**0.5 
          print('Hamiltonian o and d and ec',o,d,ec)
          ecs[itot]=ec

  # Do numerical integration of DFT and projected DFT XCs 
  EXCSL=0
  NA=0
  NB=0
  NPSPA = 0
  NPSPB = 0
  hermi=1 
  xc=ks.xc
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  print('TEH XC ',xc,hyb)

  if(not(xc=='HF,' or xc=='hf,')):
    ni = ks._numint
    ao_deriv=0
    xctype=ni._xc_type(xc)
    if xctype == 'GGA':
      ao_deriv=1
    elif xctype == 'MGGA':
      ao_deriv=1
    nao=m.nao
    tiny = 0.00000001
    make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
    make_rhpspa, nset = ni._gen_rho_evaluator(m, [PSP[0]], hermi, False, ks.grids)[:2]
    make_rhpspb       = ni._gen_rho_evaluator(m, [PSP[1]], hermi, False, ks.grids)[0]
    make_rhpas=[]
    make_rhpbs=[]
    for PP in PPs:
      make_rhpa, nset = ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[:2]
      make_rhpb       = ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0]
      make_rhpas.append(make_rhpa)
      make_rhpbs.append(make_rhpb)
    make_rhppsppas=[]
    make_rhppsppbs=[]
    for PpSPp in PpSPps:
      make_rhppsppa, nset = ni._gen_rho_evaluator(m, [PpSPp[0]], hermi, False, ks.grids)[:2]
      make_rhppsppb       = ni._gen_rho_evaluator(m, [PpSPp[1]], hermi, False, ks.grids)[0]
      make_rhppsppas.append(make_rhppsppa)
      make_rhppsppbs.append(make_rhppsppb)

    # Use gen_rho_evaluator to compute exact exchange energy densities 
    exctype = 'LDA' 
    make_exfa = ni._gen_rho_evaluator(m, [R[0]], hermi, False, ks.grids)[0]
    make_exfb = ni._gen_rho_evaluator(m, [R[1]], hermi, False, ks.grids)[0]
    make_expfa = ni._gen_rho_evaluator(m, [RF[0]], hermi, False, ks.grids)[0]
    make_expfb = ni._gen_rho_evaluator(m, [RF[1]], hermi, False, ks.grids)[0]
    make_expas=[]
    make_expbs=[]
    for RP in RPs:
      make_expa = ni._gen_rho_evaluator(m, [RP[0]], hermi, False, ks.grids)[0]
      make_expb = ni._gen_rho_evaluator(m, [RP[1]], hermi, False, ks.grids)[0]
      make_expas.append(make_expa)
      make_expbs.append(make_expb)

    for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):

        aosex=aos[0]
        if(ao_deriv<1):
          aosex = aos[0]
        print('!!!',aos.shape)

        # Density 
        rho_a = make_rhoa(0, aos, mask, xctype)
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
        rhpsp_a = make_rhpspa(0, aos, mask, xctype)
        rhpsp_b = make_rhpspb(0, aos, mask, xctype)
        if(len(rhpsp_b.shape)>1):
          rhpsp_b[0,rhpsp_b[0]<tiny]=tiny
          rhpspb = rhpsp_b[0]
        else:
          rhpsp_b[rhpsp_b<tiny]=tiny
          rhpspb = rhpsp_b
        if(len(rhpsp_a.shape)>1):
          rhpsp_a[0,rhpsp_a[0]<tiny]=tiny
          rhpspa = rhpsp_a[0]
        else:
          rhpsp_a[rhpsp_a<tiny]=tiny
          rhpspa = rhpsp_a

        NA = NA+numpy.dot(rhoa,weight)
        NB = NB+numpy.dot(rhob,weight)
        rho = (rho_a, rho_b)
 
        # LH terms 
        #aos2 = m.eval_gto("GTOval_sph",coords)
        #exfull=numpy.einsum('ri,sij,rj->r',aos2,R,aos2)
        #expf=numpy.einsum('ri,sij,rj->r',aos2,RF,aos2)
        exfull=make_exfa(0, aosex, mask, exctype)+make_exfb(0,aosex,mask,exctype)
        expf=make_expfa(0, aosex, mask, exctype)+make_expfb(0,aosex,mask,exctype)
        #print('!!!',exfull.shape)
        exfull[exfull<tiny]=tiny
        exfull=-0.5*exfull
        expf[expf<tiny]=tiny
        expf=-0.5*expf

        # SL XC 
        x=xc
        x=(x.split(','))[0] + ','
        excsl=numpy.zeros_like(rhoa)
        exsl=numpy.zeros_like(rhoa)
        ecsl=numpy.zeros_like(rhoa)
        if(xc!='hf'):
          excsl, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
          if(x=='lda,' or x=='LDA,'):
            exsl = ni.eval_xc_eff(x, (rhoa,rhob), deriv=0, xctype=xctype)[0]
          else:
            if(not(x=='hf,' or x=='HF,')):
              print('LOOK WE ARE EVALUATING EXCHANGE |%s|'%(x))
              exsl, vxc = ni.eval_xc_eff(x, rho, deriv=1, xctype=xctype)[:2]
          excsl = excsl*(rhoa+rhob) 
          exsl = exsl*(rhoa+rhob) 
          ecsl=excsl-exsl
        print('??',numpy.dot(weight,excsl),numpy.dot(weight,exsl))

        # TEST: To understand how these weights work, let's just use the exact exchange!
        #exsl = exfull 
        #excsl = exfull 

        # LH XC 
        #exslf = exsl
        #if(hyb>0.000001):
        #  exslf = exsl/(1-hyb)
        #z=(exslf/exfull)-1
        #z[z<tiny]=tiny
        #ahflh=erf(lam*z) 
        #ahflh=numpy.zeros_like(excsl) # TEST turn off local hybrid bit 
        #excsl = ahflh*exfull + (1-ahflh)*excsl
        #print('??',numpy.dot(weight,excsl))
        
        EXCSL = EXCSL+numpy.dot(excsl,weight) 

        # Projected densities and XCs 
        ngrids = coords.shape[0]
        for ip in range(len(PPs)):
          make_rhpa = make_rhpas[ip]
          make_rhpb = make_rhpbs[ip]
          rhp_a = make_rhpa(0, aos, mask, xctype)
          rhp_b = make_rhpb(0, aos, mask, xctype)
          if(len(rhp_b.shape)>0):
            rhp_b[0,rhp_b[0]<tiny]=tiny
            rhpb=rhp_b[0]
          else:
            rhp_b[rhp_b<tiny]=tiny
            rhpb=rhp_b
          if(len(rhp_a.shape)>0):
            rhp_a[0,rhp_a[0]<tiny]=tiny
            rhpa=rhp_a[0]
          else:
            rhp_a[rhp_a<tiny]=tiny
            rhpa=rhp_a
          make_rhppsppa = make_rhppsppas[ip]
          make_rhppsppb = make_rhppsppbs[ip]
          rhppspp_a = make_rhppsppa(0, aos, mask, xctype)
          rhppspp_b = make_rhppsppb(0, aos, mask, xctype)
          if(len(rhppspp_b.shape)>0):
            rhppspp_b[0,rhppspp_b[0]<tiny]=tiny
            rhppsppb=rhppspp_b[0]
          else:
            rhppspp_b[rhppspp_b<tiny]=tiny
            rhppsppb=rhppspp_b
          if(len(rhppspp_a.shape)>0):
            rhppspp_a[0,rhppspp_a[0]<tiny]=tiny
            rhppsppa=rhppspp_a[0]
          else:
            rhppspp_a[rhppspp_a<tiny]=tiny
            rhppsppa=rhppspp_a
          NPSPA = NPSPA+numpy.dot(rhppsppa,weight)
          NPSPB = NPSPB+numpy.dot(rhppsppb,weight)
          make_expa = make_expas[ip]
          make_expb = make_expbs[ip]
          expr=make_expa(0, aosex, mask, exctype)+make_expb(0,aosex,mask,exctype)
          
          expr[expr<tiny]=tiny
          expr = -0.5*expr 
          pwt = expr/exfull
          pwt2 = ((rhpa+rhpb)/(rhoa+rhob+tiny))**2
          pwt3 = ((rhppsppa+rhppsppb)/(rhpspa+rhpspb+tiny))
          # If we do this rescaling, we CANNOT have the right answer when EXC= exact exchange!  
          #pwt[pwt>1]=1
          #pwt[pwt<0]=0
          #for icoord in range(ngrids):
          #  print('FAXS %2d  %12.6f %12.6f %12.6f %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e  '%(ip,coords[icoord,0],coords[icoord,1],coords[icoord,2],pwt[icoord],pwt2[icoord],pwt3[icoord],rhoa[icoord],rhpa[icoord],rhppsppa[icoord]))
        
          pwt = pwt3
          print('!!!',ip,numpy.dot(rhoa,weight),numpy.dot(rhpspa,weight),numpy.dot(rhpa,weight),numpy.dot(rhppsppa,weight),numpy.dot(rhoa*pwt,weight),numpy.dot(exfull,weight),numpy.dot(expr,weight))
          excsls[ip]=excsls[ip]+ numpy.dot(excsl*pwt,weight)

  print('Numint Test NA,NB ',NA,NB)
  print('Numint Test NPSPA,NPSPB ',NPSPA,NPSPB)
  wts1 = 1.0/wts
  wts1 = wts1**2
  print('EX    ',EX)
  print('EXPS  ',exs)
  print('ECPS  ',ecs)
  print('EXC   ',EXCSL)
  print('EXCPS ',excsls)
  print('WEIGHTS ',wts1)
  EC = numpy.dot(ecs  ,wts1)
  EXP = numpy.dot(exs  ,wts1)
  EXCSLP= numpy.dot(excsls,wts1)

  return(EC,EXP,EXCSL,EXCSLP)


def euci(ks,hl=0,Pin=None):
  # Regenerate individual shells UCI correlation, exchange, and DFT XC energies

  # Set up 
  pAOs = ks.paos 
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # ROHF to UHF 
      print('Converting ROHF to UHF') 
      mo_a=ks.mo_coeff
      mo_b=mo_a
      e_a = ks.mo_energy 
      e_b=e_a
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec

  # Generate block-orthogonalized projected AOs 
  opAOs,SopAOs,SAOopAOs,VeeopAOs= makeOPAOs(S,pAOs,ks.VeepAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 

  # Virtual and energy weighted density matrices 
  PE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,:Na],e_a[:Na])
  PE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,:Na]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,:Nb],e_b[:Nb])
  PE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,:Nb]))
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.dot(mo_a[:,Na:],numpy.transpose(mo_a[:,Na:]))
  Pv[1] = numpy.dot(mo_b[:,Nb:],numpy.transpose(mo_b[:,Nb:]))
  PvE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],e_a[Na:])
  PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],e_b[Nb:])
  PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  # Indexing for projected natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  exs=numpy.zeros(ntot)
  excsls=numpy.zeros(ntot)
  ecs=numpy.zeros(ntot)
  ecs2=numpy.zeros(ntot)
  ecs3=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)

  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    print('Here is shell ',ishell,' overlap \n',SopAOs[ishell])
    #SopAOm=numpy.linalg.inv(SopAOs[ishell])
    SopAOm=numpy.linalg.pinv(SopAOs[ishell]) # SopAOs should be identity! 


    # This step is a timesink for large systems, as it's an AO-MO transform for
    # the entire molecule not just for the shell of interest.  The good side is
    # there's a small and asymptotically constant number of 'mos' on this shell
    # Might be smarter to compute opAO Vees from just the AOs on this atom? 
    #####Vees = O2_1to2(VeeAO,SAOopAOs[ishell],Sm)
    #tt = numpy.dot(Sm,SAOopAOs[ishell])
    #print('TEST TT ',tt.shape)
    #print('TEST2 TT ',numpy.einsum('mi,mn,nj->ij',tt,S,tt))
    #Vees2 = m.ao2mo(tt) 
    #Vees = ao2mo.restore(1,numpy.asarray(Vees2),tt.shape[1])
    Vees = VeeopAOs[ishell]
    print('TEST VEEs ',Vees.shape)
    print('VEES FIRST ',Vees[0,0,0,0])

    # Extra testing 
    #print('VEE, MO \n',Vees)
    #print('VEE, AO\n',m.intor('int2e'))

    # Density matrices in this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(Pv,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)
    PvEs=P_1to2(PvE,SAOopAOs[ishell],SopAOm)
    print('Shell ',ishell,' alpha density \n',Ps[0])
    print('Shell ',ishell,' projected 1PDM diagonal ')
    for iproj in range(Ps.shape[1]):
      print('%6.3f %6.3f '%(Ps[0,iproj,iproj],Ps[1,iproj,iproj]))

    # Loop over atomic natural orbitals in this shell 
    # The 'atomic natural orbitals' are eigenvectors of the density matrix
    # projected onto the opAOs. 
    #(vals,vecs)=numpy.linalg.eigh(Ps[0])
    (vals,vecs)=numpy.linalg.eigh(Ps[0]+Ps[1])
    # Note that the pAOs in this shell are not orthogonal. 
    print('Shell ',ishell,' alpha+beta density eigenvalues ',vals)
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj]

       # Do FCI on just this ANO 
       if(hl>0): 
         vf = V_1to2(v,numpy.transpose(SAOopAOs[ishell]),Sm)
         print('Here is shell ',ishell,' pano ',iproj,'\n',v,'\n',vf)
         print('Overlaps ',numpy.einsum('i,i->',v,v),numpy.einsum('m,mn,n->',vf,S,vf))
         noa1=numpy.dot(v,numpy.dot(Ps[0],v))
         noa2=numpy.dot(vf,numpy.dot(S,numpy.dot(P[0],numpy.dot(S,vf))))
         temp = numpy.einsum('mi,mn,n->i',mo_a[:,:Na],S,vf)
         noa3 = numpy.dot(temp,temp)
         print('Sanity tests of FCI on an ANO: ',noa1,noa2,noa3)
         q=numpy.einsum('m,n->mn',vf,vf)
         test = (numpy.dot(q,numpy.dot(S,q))-q)**2
         print('HL 1 TEST: ',numpy.einsum('ij->',test))
         qs = numpy.dot(q,S)
         print('Calling ECI from euci with hl > 0')
         teh = eci(ks=ks,QS=qs,Pin=Pin)
         ecs2[itot] = teh[0]
         ecs3[itot] = teh[1]

       # Do FCI on this ANO plus all virtual orbitals outside of the full
       # projection space, yielding something
       # analogos to the independent electron pair approximation.
#       if(hl>1): 
#         vf = V_1to2(v,numpy.transpose(SAOopAOs[ishell]),Sm)
#         q=numpy.einsum('m,n->mn',vf,vf)
#         qv = Pv[0]
#         print('q ',q.shape)
#         print('qv ',qv.shape)
#         print('S ',S.shape)
#         print('QS ',ks.QS[0].shape)
#         qva = numpy.dot(qv,numpy.dot(S,numpy.dot(ks.QS[0],qv)))
#         print('qva ',qva.shape)
#         q= q + qv - qva 
#         test = (numpy.dot(q,numpy.dot(S,q))-q)**2
#         print('HL 2 TEST: ',numpy.einsum('ij->',test))
#         qs = numpy.dot(q,S)
#         ecs3[itot]  = eci(ks=ks,QS=qs,Pin=Pin)

       #w0 = numpy.einsum('p,pq->q',v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       w0 = numpy.dot(v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = numpy.dot(w0,w0)
       wts[itot]=wt
       print('Total Weight Projection ',wt)
       print('Weight ',1/wt)

       J=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,Vees))))

       # Assemble the 2x2 Hamiltonian 
       noa=numpy.dot(v,numpy.dot(Ps[0],v))
       nob=numpy.dot(v,numpy.dot(Ps[1],v))
       nva=numpy.dot(v,numpy.dot(Pvs[0],v))
       nvb=numpy.dot(v,numpy.dot(Pvs[1],v))
       eoa=numpy.dot(v,numpy.dot(PEs[0],v))
       eob=numpy.dot(v,numpy.dot(PEs[1],v))
       eva=numpy.dot(v,numpy.dot(PvEs[0],v))
       evb=numpy.dot(v,numpy.dot(PvEs[1],v))
       print('Occupancies ',ishell,iproj,noa,nob)
       print('Virt occs   ',nva,nvb)
       print('Self energy ',J)

       # Exact exchange and DFT+U-type semilocal exchange 
       exs[itot]   = -J*(noa**2+nob**2)/2
       excsls[itot] = -J*(noa+nob)/2

       # Diagonalize the 2x2 CI Hamiltonian 
       if(noa>0.000001 and nob>0.000001 and nva>0.000001 and nvb>0.000001):
          eoa = eoa/(noa +0.00000001)
          eob = eob/(nob +0.00000001)
          eva = eva/(nva +0.00000001)
          evb = evb/(nvb +0.00000001)
          print('Proj Occ Energies    ',eoa,eob)
          print('Proj Virt Energies',eva,evb)

          # Diagonalize 
          o = J*(noa*nob*nva*nvb)**0.5 # <Phi_0|Vp|Phi_oo^vv> 
          o = numpy.maximum(o,1e-10)
          ee=(eva+evb-eoa-eob)          
          #d = (ee + J*(noa*nob+nva*nvb-noa*nvb-nob*nva))/2 # <Phi_oo^vv|Vp|Phi^oo_vv> - <Phi_0|Vp|Phi_0>
          #d = (ee + J*(nva*nvb-noa*nob))/2 # <Phi_oo^vv|Vp|Phi^oo_vv> - <Phi_0|Vp|Phi_0>
          d = ee/2 # TEST typically little difference 
          #d = 0 
          ec = d-(d**2+o**2)**0.5 
          print('Hamiltonian o and d and ec',o,d,ec)
          ecs[itot]=ec

  wts1 = 1.0/wts
  wts1 = wts1**2
  print('EUCI ECS ',ecs,ecs2,ecs3)
  print('WEIGHTS ',wts1)
  EC1 = numpy.dot(ecs  ,wts1)
  EC2 = numpy.dot(ecs2 ,wts1)
  EC3 = numpy.dot(ecs3 ,wts1)
  EX  = numpy.dot(exs  ,wts1)
  EXSL= numpy.dot(excsls,wts1)

  ### Projected CI in the full active space 
  EC4 = 0 
  if(hl>2): 
    print('Calling ECI from euci with hl > 2')
    EC4 = eci(ks,Pin=Pin)[0]
  return(EC1,EC2,EC3,EC4,EX,EXSL)


##### 
class AsFCISolver(object):
    def __init__(self):
        self.mycc = None

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        fakemol.spin=nelec[0]-nelec[1]
        nelec = numpy.sum(nelec)
        fakemol.nelectron = nelec
        fake_hf = scf.UHF(fakemol)
        print('Look we made a fake hf ')
        print('h1 shape ',h1.shape)
        #fake_hf._eri = ao2mo.restore(8, h2, norb)
        fake_hf._eri = ao2mo.restore(8, h2[0], norb) # bgj use alpha-alpha eris here 
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        fake_hf.kernel()
        print('Fake HF energy ',fake_hf.e_tot)
        self.mycc = cc.UCCSD(fake_hf)
        eris = self.mycc.ao2mo() # This is a _ChemistsERIs object not just a list of ERIs. It has a bunch of crap like Fock operators, OOOO and OVOV blocks, and other horseshit in it 
        e_corr, t1, t2 = self.mycc.kernel(eris=eris)
        return(e_corr)
        #print('Fake correlation energy ',e_corr)
        #l1, l2 = self.mycc.solve_lambda(t1, t2, eris=eris)
        #print('Fake total energy ',self.mycc.e_tot)
        #e_tot = self.mycc.e_tot + ecore
        #return e_tot, CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])

    def make_rdm1(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm1 = self.mycc.make_rdm1(t1, t2, l1, l2, ao_repr=True)
        return dm1

    def make_rdm12(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm2 = self.mycc.make_rdm2(t1, t2, l1, l2, ao_repr=True)
        return self.make_rdm1(fake_ci, norb, nelec), dm2

    def spin_square(self, fake_ci, norb, nelec):
        return 0, 1
class CCSDAmplitudesAsCIWfn:
    def __init__(self, cc_amplitues):
        self.cc_amplitues = cc_amplitues

def euci5(ks,Pin=None,hl=0,funcsets=None):
  # September 23 2025 
  # While projecting Vee onto a single function |phi> ensures that the CI has a 2x2
  # active space, the particular choice of active space we've made is not
  # variationally optimal. We choose the s-spin virtual orbital |psivs> to maximize
  # |<psivs|phi>|, regardless of whether this includes extremely high-energy
  # virtual orbitals. In this variant, we'll test other choices

  # October 27 2025, funcsets offers CI on multiple projection functions using the
  # optimized MOs from single projection functions. Enables e.g. CAS(12,12) on
  # pairs of N atoms in N100. 

  # March 2026, return the projections Q from both sets for subsequent use 

  # Set up 
  Q1s=[]
  Q2s=[]
  pAOs = ks.paos 
  m = ks.mol
  (Na,Nb)=m.nelec
  nao = m.nao
  ni = ks._numint
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  N=(S.shape)[0]
  (vals,vecs) = linalg.eigh(S)
  Smhalf = numpy.zeros((N,N))
  for i in range(N):
    if(vals[i]>0.00000001):
      Smhalf[i,i] = ((vals[i]).real)**(-0.5)
      
  # Build UHF-indexed 1PDM, mos, mo energies 
  if(len(P.shape)<3): # RHF to UHF 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # ROHF to UHF 
      raise Exception('euci4 NYI for ROHF')
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec
  print('NUMTEST ',numpy.einsum('sij,ij->s',P,S))

  # Generate the orthogonalized projected AOs with which to build the projected
  # atomic natural orbitals. This part could be done just once, not in every
  # SCF cycle. 
  opAOs,SopAOs,SAOopAOs,VeeopAOs= makeOPAOs(S,pAOs,ks.VeepAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 
  dum1,dum2,dum3,VeeRSopAOs= makeOPAOs(S,pAOs,ks.VeeRSpAOs)

  # Energy weighted density matrices 
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.dot(mo_a[:,Na:],numpy.transpose(mo_a[:,Na:]))
  Pv[1] = numpy.dot(mo_b[:,Nb:],numpy.transpose(mo_b[:,Nb:]))
  PE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,:Na],e_a[:Na])
  PE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,:Na]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,:Nb],e_b[:Nb])
  PE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,:Nb]))
  PvE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],e_a[Na:])
  PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],e_b[Nb:])
  PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  # Indexing for projected atomic natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  exs=numpy.zeros(ntot)
  ecs=numpy.zeros(ntot)
  #cvals=numpy.zeros((ntot,2))
  eovvals=numpy.zeros((ntot,4))
  ecps=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)
  # Save the projections Q and the optimized projected MOs entering the CI 
  kepts = numpy.zeros(ntot)
  Js = numpy.zeros(ntot)
  phis=numpy.zeros((ntot,N))
  oas=numpy.zeros((ntot,N))
  vas=numpy.zeros((ntot,N))
  obs=numpy.zeros((ntot,N))
  vbs=numpy.zeros((ntot,N))

  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    SopAOm=numpy.linalg.pinv(SopAOs[ishell])
    Vees = VeeopAOs[ishell]

    # Project density matrices into this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(P,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)

    # Build atomic natural orbitals in this shell 
    (vals,vecs)=numpy.linalg.eigh(Ps[0]+Ps[1])
    print('Shell ',ishell,' total density eigenvalues ',vals)
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj] 
       vv = numpy.outer(v,v)
       v2=numpy.dot(v,SAOopAOs[ishell].T)
       # v2(mu) is <chi_mu|phi> 
       v3=numpy.dot(Sm,numpy.dot(v,SAOopAOs[ishell].T))
       print('V3 test',numpy.dot(v3,numpy.dot(S,v3)))

       # Weight is |phi> overlap with ALL projected atomic orbitals 
       w0 = numpy.dot(v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = 1.0/numpy.dot(w0,w0)
       wts[itot]=wt
       print('Total Weight',wt)

       # Self-energy 
       J=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,Vees))))

       # Here we choose the transformed occ and virt orbitals entering the 2x2 CI

       # Occupation numbers and projections 
       noa=numpy.dot(v,numpy.dot(Ps[0],v))
       nob=numpy.dot(v,numpy.dot(Ps[1],v))
       nva=numpy.dot(v,numpy.dot(Pvs[0],v))
       nvb=numpy.dot(v,numpy.dot(Pvs[1],v))
       exs[itot]   = -J*(noa**2+nob**2)/2

       # etaoa(i) is <phi|psi_i> computed as sum(mu) <phi|chi_mu><chi_mu|psi_i>
       etaoa = numpy.einsum('m,ma->a',v2,mo_a[:,:Na])
       etaob = numpy.einsum('m,ma->a',v2,mo_b[:,:Nb])
       etava = numpy.einsum('m,ma->a',v2,mo_a[:,Na:])
       etavb = numpy.einsum('m,ma->a',v2,mo_b[:,Nb:])
       print('euci5 proj test ',noa,nob,numpy.dot(etaoa,etaoa),numpy.dot(etaob,etaob))

       # Here we initialize the four unitary transform matrices moa,mob...
       # and build the matrices used in their updates 
       # eoas(i,j) = eorb(i) delta(i,j)
       eoas= numpy.diag(e_a[:Na])
       eobs= numpy.diag(e_b[:Nb])
       evas= numpy.diag(e_a[Na:])
       evbs= numpy.diag(e_b[Nb:])
       # etaoas(i,j) is <psi_i|phi><phi|psi_j> 
       etaoas = numpy.einsum('i,j->ij',etaoa,etaoa)
       etaobs = numpy.einsum('i,j->ij',etaob,etaob)
       etavas = numpy.einsum('i,j->ij',etava,etava)
       etavbs = numpy.einsum('i,j->ij',etavb,etavb)
       moa = -J   *nob*nva*nvb* etaoas
       mob = -J*noa   *nva*nvb* etaobs
       mva = -J*noa*nob   *nvb* etavas
       mvb = -J*noa*nob*nva   * etavbs 
       Ec=0
       cval=0

       EcP=0

       if(Na>0 and Nb>0 and noa>0.001 and nob>0.001 and noa<0.999 and nob<0.999):
       #if(True): # Use all so that we can do the other stuf 
         for itr in range(5):

           # Diagonalize the Fock-like matrices in the MO basis and choose the
           # new eigenvectors
           (va,ve) = linalg.eigh(moa)
           doa = ve[:,numpy.argmin(va)]
           (va,ve) = linalg.eigh(mob)
           dob = ve[:,numpy.argmin(va)]
           (va,ve) = linalg.eigh(mva)
           dva = ve[:,numpy.argmin(va)]
           (va,ve) = linalg.eigh(mvb)
           dvb = ve[:,numpy.argmin(va)]

           # Build the energy and the new Fock-like matrices 
           noa=numpy.dot(doa,etaoa)**2
           nob=numpy.dot(dob,etaob)**2
           nva=numpy.dot(dva,etava)**2
           nvb=numpy.dot(dvb,etavb)**2 
           eoa = numpy.dot(doa,numpy.dot(eoas,doa))
           eob = numpy.dot(dob,numpy.dot(eobs,dob))
           eva = numpy.dot(dva,numpy.dot(evas,dva))
           evb = numpy.dot(dvb,numpy.dot(evbs,dvb))
           eovvals[itot,:]=[eoa,eob,eva,evb]
           print('Proj N and J: ',iproj,noa,nob,nva,nvb,J)
           print('Proj Occ and Virt: ',eoa,eob,eva,evb)
           o = noa*nob*nva*nvb
           Del = (eva+evb-eoa-eob)/2 
           print('O and Del: ',o,Del)
           denom2 = (J**2*o+(Del+(Del**2+J**2*o)**0.5)**2)**0.5
           if(denom2<0.0000001):
             denom2 =0.0000001
           c0val =-(Del+(Del**2+J**2*o)**0.5)/denom2
           c1val = o**0.5*J/denom2 
           denom = J**2*o+Del**2
           if(denom<0.0000001):
             denom =0.0000001
           denom = denom**(-0.5)
           Ec = Del - (J**2*o + Del**2)**0.5
           Eco = -J**2/2*denom
           EcDel = 1-Del*denom
           moa =  Eco   *nob*nva*nvb* etaoas - EcDel*eoas/2
           mob =  Eco*noa   *nva*nvb* etaobs - EcDel*eobs/2
           mva =  Eco*noa*nob   *nvb* etavas + EcDel*evas/2
           mvb =  Eco*noa*nob*nva   * etavbs + EcDel*evbs/2

           # Restrained deominator perturbation theory 
           tau= ks.mp2lam
           if(tau is None):
             tau = 2.39
           Delta = max(tau,eva+evb-eoa-eob)
           Fac = 1/Delta
           EcP = -2*J**2*o*Fac
           print('+++ %2d %12.6f %12.6f %12.6f %8.4f %8.4f %8.4f %8.4f '%(itr,Ec,EcP,Del,noa,nob,nva,nvb))
           
         # Do these things only if we did CI on this state
         #cvals[itot,:]=[c0val,c1val]
         oas[itot,:] = numpy.einsum('mi,i->m',mo_a[:,:Na],doa) #<chi_mu|psi_oa> = <chi_mu|psi_i><psi_i|psi_oa>
         vas[itot,:] = numpy.einsum('mi,i->m',mo_a[:,Na:],dva)
         obs[itot,:] = numpy.einsum('mi,i->m',mo_b[:,:Nb],dob)
         vbs[itot,:] = numpy.einsum('mi,i->m',mo_b[:,Nb:],dvb)
         print('OAS TEST 1',numpy.dot(oas[itot],numpy.dot(S,oas[itot])))
         print('VAS TEST 1',numpy.dot(vas[itot],numpy.dot(S,vas[itot])))
         print('OAS TEST 2',numpy.dot(oas[itot],numpy.dot(S,mo_a[:,:Na])))
         print('VAS TEST 2',numpy.dot(vas[itot],numpy.dot(S,mo_a[:,Na:])))
         ecs[itot]=Ec
         ecps[itot]=EcP
       # Do these things even if there's no CI in this state 
       kepts[itot] = 1 
       Js[itot] = J
       phis[itot,:] = v3
       tehpao = numpy.transpose(numpy.array(phis[itot:itot+1,:]))
       Q = pao_proj(ks,pAOs=[tehpao],doret=True)
       Q1s.append(Q)

  # Correlation in sets of functions. (If not already set we'll use all pairs of functions) 
  EC2 = 0
  ECP2 = 0
  wts2=[]
  ecs2=[]
  ecps2=[]
  if(hl>1 and sum(kepts)>0.00000001):

    # build the state overlaps to use in weights weights  [sum kl] <phi_i,phi_j|phi_k,phi_l> 
    wts1=numpy.zeros((ntot,ntot))
    for iproj in range(ntot):
      if(kepts[iproj]>0):
        for jproj in range(ntot): 
          if(kepts[jproj]>0):
            val= numpy.dot(phis[iproj],numpy.dot(S,phis[jproj]))
            wts1[iproj,jproj] = val
    print('wts1 \n',wts1)

    if(funcsets is None):
      funcsets = []
      for iproj in range(ntot):
        for jproj in range(0,iproj): # Loop over all different pairs once 
         if(kepts[iproj]>0 and kepts[jproj]>0):
           funcsets.append([iproj,jproj])
    print('funcsets \n',funcsets)

    for iset in range(len(funcsets)):
      wt=1
      for jset in range(len(funcsets)): # overlap with all other sets 
        if(jset != iset):
          val = 1 
          for iproj in funcsets[iset]:
            for jproj in funcsets[jset]:
              val = val*(wts1[iproj,jproj])**2
          wt = wt + val
      print('look the 1/wt is ',wt)
      if(wt>1e-6):
        wt=1.0/wt
      else:
        wt=0
      tehpaos0=[]
      for iproj in funcsets[iset]:
        if(kepts[iproj]>0):
          tehpaos0.append(phis[iproj])
      tehpaos = numpy.transpose(numpy.array(tehpaos0))
      # TEST 
      print('Overlap of paos in ',funcsets[iset])
      Stest = numpy.dot(tehpaos.T,numpy.dot(S,tehpaos))
      print(Stest)
      Q = pao_proj(ks,pAOs=[tehpaos],doret=True)
      Q2s.append(Q)
      QS = numpy.dot(Q,S)
      print('Multi-function CI ',funcsets[iset],' wt ',wt,' test ',numpy.einsum('ij->',numpy.dot(QS,Q)-Q))
      QMOoa = numpy.zeros((nao,nao))
      QMOob = numpy.zeros((nao,nao))
      QMOva = numpy.zeros((nao,nao))
      QMOvb = numpy.zeros((nao,nao))
      for iproj in funcsets[iset]:
        if(kepts[iproj]>0):
          QMOoa = QMOoa + numpy.einsum('m,n->mn',oas[iproj],oas[iproj]) 
          QMOva = QMOva + numpy.einsum('m,n->mn',vas[iproj],vas[iproj]) 
          QMOob = QMOob + numpy.einsum('m,n->mn',obs[iproj],obs[iproj]) 
          QMOvb = QMOvb + numpy.einsum('m,n->mn',vbs[iproj],vbs[iproj]) 
      QMOs = [QMOoa,QMOva,QMOob,QMOvb]
      print('Calling ECI from euci5 for multi-function CI with functions ',funcsets[iset])
      ECij,EMP2ij = eci(ks,QS=QS,QMOs=QMOs,Pin=Pin,hl=hl-2)
      
      # Subtract off term 
      DECij= ECij 
      DEMP2ij= EMP2ij 
      for iproj in funcsets[iset]:
        if(kepts[iproj]>0):
          ww = 0 
          for jproj in funcsets[iset]:
             ww = ww+wts1[iproj,jproj]**2
          #print('Correcting with ',iproj,ecs[iproj],ww)
          DECij = DECij - ecs[iproj]/ww
          DEMP2ij = DEMP2ij - ecps[iproj]/ww
      print('Multi-function EC ',funcsets[iset],ECij,DECij,EMP2ij,DEMP2ij)
      wts2.append(wt)
      ecs2.append(DECij)
      ecps2.append(DEMP2ij)
    print('EUCI5 multi-function corrections ECS2',ecs2)
    print('EUCI5 multi-function corrections ECPS2',ecps2)
    print('WTS2   ',wts2)
    EC2 = numpy.dot(wts2,ecs2)
    ECP2 = numpy.dot(wts2,ecps2)

  print('EUCI5 EXS ',exs)
  print('EUCI5 ECS after',ecs)
  print('ECPTPS',ecps)
  print('WTS   ',wts)
  EX = numpy.dot(exs  ,wts)
  EC = numpy.dot(ecs  ,wts)
  ECP = numpy.dot(ecps  ,wts)
  print('EUCI5 EC(1), ECMP2(1), DEC(2), ECMP2(2) ',EC,ECP,EC2,ECP2)
  return (EC,ECP,EX,EC+EC2,ECP+ECP2,[Q1s,Q2s],[wts,wts2])



##### Utility function for Gaussian 
names=['Dummy','H',          'He',
'Li','Be',   'B','C','N','O','F','Ne',
'Na','Mg',   'Al','Si','P','S','Cl','Ar',
'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
 
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
    b = 'def2svp'
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
  
