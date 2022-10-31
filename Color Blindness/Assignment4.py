import numpy as np
import scipy as sp
from scipy import stats

def loadLastCol(filename):
    LastCol = np.loadtxt(filename, dtype=str)
    LastCol = ''.join(LastCol)
    return LastCol

def loadRefSeq(filename):
    RefSeq_st = np.loadtxt(filename, dtype=str)
    RefSeq = ''.join(RefSeq_st[1:])
    l = ['$' for i in range(0,100)]
    l = ''.join(l)
    return RefSeq + l

def loadReads(filename):
    reads = np.loadtxt(filename, dtype=str)
    Reads = []
    for read in reads:
        if 'N' in read:
            new_read = read.replace('N', 'A')
            Reads.append(new_read)
        else:
            Reads.append(read)      
    return Reads

def loadMapToRefSeq(filename):
    MapToRefSeq = np.loadtxt(filename, dtype=int)
    return MapToRefSeq

RedExonPos = np.array([
    [149249757, 149249868], # R1
    [149256127, 149256423], # R2
    [149258412, 149258580], # R3
    [149260048, 149260213], # R4
    [149261768, 149262007], # R5
    [149264290, 149264400]  # R6
    ])
"""
2. Green Exon Locations
"""
GreenExonPos = np.array([
    [149288166, 149288277], # G1
    [149293258, 149293554], # G2
    [149295542, 149295710], # G3
    [149297178, 149297343], # G4
    [149298898, 149299137], # G5
    [149301420, 149301530]  # G6
    ])

def MatchReadToLoc(read):
    
    div=len(read)//3
    print(len(read))
    seq=[]
    seq.append(read[:div])
    seq.append(read[div:2*div])
    seq.append(read[2*div:])
    match_pos=[]
    
    def Rank(i,char):
        N=1000
        ind=i//N
        r=LastCol[ind*N:i].count(char)
        overall_rank=r+delta_arr[char][ind*N]
        return overall_rank
    
    for j in range(len(seq)):
        seqi=seq[j]
        char=seqi[-1]
        start_ch=first_col[char]
        end_ch=first_col[char+'end']
        length=end_ch-start_ch
        for char in reversed(seqi[:-1]):
            x=Rank(int(start_ch),char)
            y=Rank(int(end_ch),char)
            length=y-x
            if length==0:
                break
            start_ch= first_col[char]+x
            end_ch=first_col[char]+y
        if length:
            poss_match_ind=np.arange(start_ch,end_ch,1)
            lp=Map[poss_match_ind]-j*div
            for loc in lp:
                match_pos.append(loc)
    return np.unique(match_pos)

def WhichExon(positions):
    R= np.array([0.,0.,0.,0.,0.,0.])
    G= np.array([0.,0.,0.,0.,0.,0.])
    
    if len(positions)>0:
        for position in positions:
            mismatch=0
            for i in range(len(read)):
                if (RefSeq[position+i]!=read[i]):
                    mismatch+=1 
            if mismatch<3:                
                signR = np.sign(RedExonPos-position)
                signG = np.sign(GreenExonPos-position)
                signR = signR[:,0]*signR[:,1]
                signG = signG[:,0]*signG[:,1]
                
                R += np.array([0.5 if i<0 else 0 for i in signR])
                G += np.array([1 if i<0 else 0 for i in signG])
                
        factor=np.sum(R)+np.sum(G)
        prob=(1/factor) if factor>0 else 1
        R=prob*R
        G=prob*G
    return np.concatenate((R,G),axis=0)

def ComputeProb(ExonMatchCounts):

    green_red=ExonMatchCounts[1:5]/ExonMatchCounts[7:11]
    case1=[0.5,0.5,0.5,0.5]
    case2=[1,1,0,0]
    case3=[0.33,0.33,1,1]
    case4=[0.33,0.33,0.33,1]
    
    p0= sp.stats.multivariate_normal.pdf(green_red, mean=case1)
    p1= sp.stats.multivariate_normal.pdf(green_red, mean=case2)
    p2= sp.stats.multivariate_normal.pdf(green_red, mean=case3)
    p3= sp.stats.multivariate_normal.pdf(green_red, mean=case4)
    
    s=sum([p0,p1,p2,p3])
    return [p0/s, p1/s, p2/s, p3/s]

def BestMatch(ListProb):
    return np.argmax(ListProb) 

def Rank_delta(LastCol):
    last_col_A=[0]*len(LastCol)
    last_col_C=[0]*len(LastCol)
    last_col_G=[0]*len(LastCol)
    last_col_T=[0]*len(LastCol)
    
    for i in range(len(LastCol)):
        if LastCol[i]=='A':
            last_col_A[i]=1
        if LastCol[i]=='C':
            last_col_C[i]=1
        if LastCol[i]=='G':
            last_col_G[i]=1
        if LastCol[i]=='T':
            last_col_T[i]=1
              
    N=1000
    delta_A={0:0}
    delta_T={0:0}
    delta_C={0:0}
    delta_G={0:0}
    
    for i in range(len(LastCol)//N):
        delta_A[(i+1)*N]=delta_A[i*N]+np.sum(last_col_A[i*N:(i+1)*N])
        delta_T[(i+1)*N]=delta_T[i*N]+np.sum(last_col_T[i*N:(i+1)*N])
        delta_C[(i+1)*N]=delta_C[i*N]+np.sum(last_col_C[i*N:(i+1)*N])
        delta_G[(i+1)*N]=delta_G[i*N]+np.sum(last_col_G[i*N:(i+1)*N])
            
    return {'A':delta_A, 'C':delta_C, 'T':delta_T, 'G':delta_G}

def reverse_complement(s):
    s=s[::-1] 
    bases = list(s)
    for i in range(len(bases)):
        if bases[i]=='A':
            bases[i]=='T'
        elif bases[i]=='C':
            bases[i]=='G'
        elif bases[i]=='G':
            bases[i]=='C'
        elif bases[i]=='T':
            bases[i]=='A'
    return ''.join(bases)

if __name__ == "__main__":
    
    #load all the data files
    LastCol = loadLastCol("../data/chrX_last_col.txt") # loads the last column
    RefSeq = loadRefSeq("../data/chrX.fa") # loads the reference sequence
    Reads = loadReads("../data/reads") # loads the reads
    Map = loadMapToRefSeq("../data/chrX_map.txt") # loads the mapping to the reference sequence

    #compute
    delta_arr=Rank_delta(LastCol)
    Ac = LastCol.count('A')
    Cc = LastCol.count('C')
    Gc = LastCol.count('G')
    Tc = LastCol.count('T')
    first_col={'A':0,'Aend':Ac-1,'C':Ac,'Cend':Ac+Cc-1,'G':Ac+Cc,'Gend':Ac+Cc+Gc-1,'T':Ac+Cc+Gc,'Tend':Ac+Cc+Gc+Tc-1,'$':len(LastCol)-1}
    ExonMatchCounts = np.zeros(12) # initialize the counts for exons
    for read in Reads: # update the counts for exons
        positions = MatchReadToLoc(read) # get the list of potential match locations
        if len(positions)==0:
            read=reverse_complement(read)
            positions = MatchReadToLoc(read)
            
        ExonMatchCounts += WhichExon(positions) # update the counts of exons, if applicable
    ListProb = ComputeProb(ExonMatchCounts) # compute probabilities of each of the four configurations
    MostLikely = BestMatch(ListProb) # find the most likely configuration
    print("Configuration %d is the best match"%MostLikely)