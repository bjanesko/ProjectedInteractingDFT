from pyscf import scf,gto,dft,cc
import numpy ,scipy, sys, os.path 
from scipy import linalg
import pdft
from pdft.projwork import new_epzlh,get_ehxc, euci3,euci5, eci ,  build_proj, readChk


if __name__=='__main__':
    tehFile=sys.argv[1]
    if(os.path.isfile(tehFile)):

        # Pi system atoms from GaussView 
        fragats1=[]
        fragats2=[]
        for i in range(24):
            fragats1.append(i)
            fragats2.append(i+24)
        fragats=[fragats1,fragats2]

        # Get spin density used for fragments and set up fragments 
        mhighspin,Phighspin,mo_coeff_highspin,aorbhighspin,borbhighspin=readChk('F23.fchk')
        (Na,Nb)=mhighspin.nelec
        fullbasis=mhighspin.basis
        fragorbs=mo_coeff_highspin[0,:,Nb:Na]
        nmon = 2
        funcsets=[[0,1]] # Do projected full CI with Vee projected onto pairs of fragments
        fragnums=[1,1]
        fragments=[fragats,fragnums,fragorbs]

        # Read the molecule 
        m,P,mo_coeff,aorb,borb = readChk(tehFile)
        fullbasis = m.basis 
        Na,Nb=m.nelec
        S = m.intor_symmetric('int1e_ovlp')
        Na2,Nb2=numpy.einsum('sij,ij->s',P,S)
        print('Molecule electrons ',Na,Nb,Na2,Nb2)
        if( (Na-Na2)**2>0.2):
          sys.exit('Your number of electrons is bad')
        NAO = m.nao
        NMO = len(aorb)
        nat = m.natm
        mo_occ=numpy.zeros((2,NMO))
        mo_occ[0,:Na]=1
        mo_occ[1,:Nb]=1
        mo_energy = numpy.zeros((2,NMO))
        mo_energy[0]=aorb
        mo_energy[1]=borb

        # Set up a projected DFT initialized with the HF orbitals, energies, etc 
        md=pdft.UKS(m,xc='hf,',phyb=[0],paos='SpinAOs')
        md.fragments=fragments
        md.allc=1
        md.addMP2=False 
        md.lhlam=1
        md.mp2lam = 2.39
        md.mo_occ=mo_occ
        md.mo_energy=mo_energy
        md.mo_coeff=mo_coeff
        build_proj(md)

        # Generate the HF energy
        Eother,EXHF = get_ehxc(md,P=P,xc='hf,')[:2]
        EHF=Eother+EXHF
        print('SCF Done: E = %12.6f Hartree HF '%(EHF))

        # Generate the correlation energies and optimized orbitals using PiFCI+HF 
        EXC2,VXC,EXCI,ECCI,ECP2,ESLCI = euci3(md,xc='hf,',Pin=P,addX=True,addMP2=False,stype=3)
        EC5,ECMP25,EX5,EC52,ECMP252,Qs,wts=euci5(md,Pin=P,hl=2,funcsets=funcsets)
        ECS=[EC5,EC52]
        print('PiFCI projected exchange energies ',EXCI,EX5)
        print('PiFCI projected correlation energies ',ECCI,EC5,EC52)

        # Test PiFCI-HF with and without PBE correlation 
        for xc in ('hf,','hf,pbe'):# ,'.25*hf+.75*pbe,pbe','pbe,pbe'):
          md.xc=xc
          omega, alpha, hyb = md._numint.rsh_and_hybrid_coeff(xc, spin=md.mol.spin) # Get fraction HFX 

          # Use epzlh to generate DFT and projected PiHF+DFT energies for
          # PiFCI1+DFT-maxoverlap, PiFCI1+DFT, and PiFCI2+DFT
          Eother,EX,EXP, EXSL,EXPSL,ECSL,ECPSL = new_epzlh(md,P=P)
          EXC = hyb*EX +(1-hyb)*EXSL +ECSL
          EXCP= hyb*EXP+(1-hyb)*EXPSL+ECPSL
          EXPHF = Eother+EXC+(EXP-EXCP)
          print('SCF Done: E = %12.6f Hartree DFT %s '%(Eother+EXC,xc))
          print('SCF Done: E = %12.6f Hartree PiHF+DFT %s '%(EXPHF ,xc))
          print('SCF Done: E = %12.6f Hartree PiFCI+DFT-maxoverlap %s '%(EXPHF+ECCI,xc))
          print('SCF Done: E = %12.6f Hartree PiFCI+DFT %s '%(EXPHF+EC5,xc))
          for typ in range(2):
            EXP1=0
            EXPSL1=0
            ECPSL1=0
            for i in range(len(Qs[typ])):
                Q=Qs[typ][i]
                QS=numpy.dot(Q,S)
                SQ=numpy.dot(S,Q)
                md.QS[0]=QS
                md.SQ[0]=SQ
                Eother,EX,EXP, EXSL,EXPSL,ECSL,ECPSL = new_epzlh(md,P=P)
                EXP1 = EXP1 + EXP*wts[typ][i]
                EXPSL1 = EXPSL1 + EXPSL*wts[typ][i]
                ECPSL1 = ECPSL1 + ECPSL*wts[typ][i]
            EXCP1= hyb*EXP1+(1-hyb)*EXPSL1+ECPSL1
            EXPHF1 = Eother+EXC+(EXP1-EXCP1)
            print('SCF Done: E = %12.6f Hartree PiHF%d+DFT %s '%(EXPHF1,typ+1,xc))
            print('SCF Done: E = %12.6f Hartree PiFCI%d+DFT %s '%(EXPHF1+ECS[typ],typ+1,xc))

