import numpy as np
import matplotlib.pyplot as plt
# To remove Instruction matplotlib.pyplot commnet PlotOut Function And Line:578
import scipy.optimize as sp
from datetime import datetime
import pandas as pd

def get_times(data):
    intervals = []
    df = pd.DataFrame(data)
    for x in range(df.shape[0]):
        d1 = datetime(df[0][x],df[1][x],df[2][x],df[3][x],df[4][x])
        d2 = datetime(df[0][0],df[1][0],df[2][0],df[3][0],df[4][0])
        duration = d1 - d2
        intervals.append(duration.days + duration.seconds/86400)
    return np.array([x-intervals[0] for x in intervals])

def get_oppositions(df):
    opp = df[5] * 30 + df[6] + df[7] / 60 + df[8] / (60 * 60)
    return opp

# Q1
def MarsEquantModel(c, r, e1, e2, z, s, time, oppositions):
    oppositions=np.array(oppositions).T 
    C_dis = 1
    cx = C_dis * np.cos(np.radians(c))
    cy = C_dis * np.sin(np.radians(c))
    ex = e1 * np.cos(np.radians(e2 +z))
    ey = e1 * np.sin(np.radians(e2 +z))
    Dotlinef = (time * s + z) % 360

    X_circle = np.array([np.nan for x in range(0,np.shape(Dotlinef)[0])])
    Y_circle = np.array([np.nan for x in range(0,np.shape(Dotlinef)[0])])

    for k in range(0, np.shape(Dotlinef)[0]): 
        tanValue = np.tan(np.radians(Dotlinef[k]))
        a = (1 + tanValue ** 2)
        b = -2 * cx + 2 * tanValue * (ey - cy - ex * tanValue)
        c = (ey - cy - ex * tanValue) ** 2 + cx ** 2 - r ** 2
        roots = np.roots([a,b,c])
        x1 = roots[0]
        x2 = roots[1]
        y1 = ey + (x1 - ex) * tanValue
        y2 = ey + (x2 - ex) * tanValue
        if 0 <= Dotlinef[k] <= 90 or 270 <= Dotlinef[k] <= 360:
            X_circle[k] = x1 if x1 >= 0 else x2
            Y_circle[k] = y1 if x1 >= 0 else y2
        else:
            X_circle[k] = x1 if x1 <= 0 else x2
            Y_circle[k] = y1 if x1 <= 0 else y2
    Cal_angle = np.degrees(np.arctan2(Y_circle,X_circle))
    Act_Angle1 = np.array([i if i <= 180  else i-360 for i in oppositions])
    errors = np.subtract(Cal_angle,Act_Angle1)
    return errors, np.max(np.absolute(errors))
# End Q1

def intialValue(rf, sf, time, oppositions): #initial tuning
        def maxerror(cf, e1f, e2f, zf):
            Er, MEr = MarsEquantModel(cf, rf, e1f, e2f, zf, sf, time, oppositions)
            return MEr
        cf = 10
        e1f = 1
        e2f = 10
        for i in range(0, 3):
            z1 = np.linspace(0, 360, 360)
            zerror = np.array([maxerror(cf, e1f, e2f, z1[i]) for i in range(0, z1.shape[0])])
            zf = z1[zerror.argmin()]
            e2f1 = np.linspace(zf, 360, 360)
            e2ferror = np.array([maxerror(cf, e1f, e2f1[i], zf) for i in range(0, e2f1.shape[0])])
            e2f = e2f1[e2ferror.argmin()]
            cf1 = np.linspace(0, 360, 360)
            cferror = np.array([maxerror(cf1[i], e1f, e2f, zf) for i in range(0, cf1.shape[0])])
            cf = cf1[cferror.argmin()]
            e1f1 = np.linspace(0, .5 * rf, 300)
            e1ferror = np.array([maxerror(cf, e1f1[i], e2f, zf) for i in range(0, e1f1.shape[0])])
            e1f = e1f1[e1ferror.argmin()]
        return cf, e1f, e2f, zf

# Q2
def bestOrbitInnerParams(rf,sf,time,oppositions):

    def maxerror(x):
        c, e1, e2, z = x
        Er, MEr = MarsEquantModel(c, rf, e1, e2, z, sf, time, oppositions)
        return MEr

    x0i=[intialValue(rf,sf,time,oppositions)] 
    x0=x0i
    result = sp.minimize(maxerror,x0,method='Nelder-Mead', options={'xatol' : 1e-5 ,'disp':False, 'return_all' :False})
    cf, e1f, e2f, zf = result.x
    Er, MEr = MarsEquantModel(cf, rf, e1f, e2f, zf, sf, time, oppositions)

    return cf, e1f, e2f, zf, Er, MEr
# End Q2

# Q3
def bestS(rf, time, oppositions):
    Timep=687
    Si = np.array([360/(Timep-1),360/(Timep+1)])
    Precision_Control=20
    for i in range(0,3):
        dis_s=np.linspace(Si[0],Si[1],Precision_Control)
        MErf=np.array([np.nan for i in range(0,Precision_Control)])
        def maxerrors(sf):
            cf1, e1f1, e2f1, zf1, Erf1, MErf1 = bestOrbitInnerParams(rf, sf, time, oppositions)
            return MErf1
        for i in range(0,Precision_Control):
            MErf[i] = maxerrors(dis_s[i])
        Si=[dis_s[MErf.argmin() if not(MErf.argmin()) else MErf.argmin()-1],dis_s[MErf.argmin() if (MErf.argmin() +1)==MErf.shape[0] else MErf.argmin() + 1]]
        OptS=dis_s[MErf.argmin()]
    cf, e1f, e2f, zf, Erf, MErf = bestOrbitInnerParams(rf, OptS , time, oppositions)
    return OptS,Erf,MErf
#End Q3

# Q4
def bestR(sf,time,oppositionsf):
    C_dis = 1
    oppositions = np.array(oppositionsf).T
    int_rf = 5

    Loop_Value = 20 
    error_change=np.array([False for i in range(0, Loop_Value)])
    Mer=np.array([360 for i in range(0,Loop_Value)])
    r_vari=np.array([int_rf for i in range(0,Loop_Value)])
    last_MEr1=0
    inc_fact=.2
    Error=True
    New_rf = int_rf
    Count_Error=1
    while (Error==True and 0<New_rf<20 and Count_Error<=Loop_Value):
        error_change = np.array([False for i in range(0, Loop_Value)])
        while(np.unique(error_change, return_counts=True)[1][0] > (int((Loop_Value)/2) +3) ):
            cf, e1f, e2f, zf, Er1f, MEr1=bestOrbitInnerParams(New_rf,sf,time,oppositionsf)
            error_tol= MEr1 - last_MEr1
            last_MEr1=MEr1
            r_vari=np.append(r_vari,New_rf)
            r_vari=np.delete(r_vari,0)
            Mer = np.append(Mer, MEr1)
            Mer = np.delete(Mer, 0)
            error_change= np.append(error_change, True if error_tol >= 0 else False)
            error_change=np.delete(error_change, 0)
            R_Check = r_vari[-4:] - r_vari[-5:-1]
            if (np.any(R_Check >= 5) or np.all(R_Check < 0)):
                Count_Error +=1
                Error = True
                inc_fact += .2
                r_vari = np.delete(r_vari, -1)
                Mer = np.delete(Mer, -1)
                r_vari = np.insert(r_vari,0,int_rf)
                Mer = np.insert(Mer,0,360)
                break
            else:
                Error = False
            cx = C_dis * np.cos(np.radians(cf))
            cy = C_dis * np.sin(np.radians(cf))
            ex = e1f * np.cos(np.radians(e2f+zf))
            ey = e1f * np.sin(np.radians(e2f +zf))
            Dotlinef = (time * sf + zf) % 360
            X_Line = (ey-ex*np.tan(np.radians(Dotlinef))) / (np.tan(np.radians(oppositions))-np.tan(np.radians(Dotlinef)))
            Y_Line = X_Line * np.tan(np.radians(oppositions))
            dis_C = np.sqrt((X_Line - cx) ** 2 + (Y_Line - cy) ** 2)
            New_rf = np.mean(dis_C)
        New_rf = int_rf + inc_fact
    rf=r_vari[Mer.argmin()]

    cf, e1f, e2f, zf, Erf, MEr = bestOrbitInnerParams(rf, sf, time, oppositionsf)
    return rf, Erf, MEr
# End Q4

# Q5
def bestMarsOrbitParams(time, oppositions):
    rf=3
    sf=360/687
    MEr1f=1
    error_change=np.array([0,0,0,0])
    last_Mer=1
    while(MEr1f>(4/60)):
        rf,Er1f,MEr1f=bestR(sf,time,oppositions)
        error_change=np.append(error_change,MEr1f-last_Mer)
        last_Mer=MEr1f
        sf, Er, MEr2f = bestS(rf,time,oppositions)
        error_change = np.append(error_change, MEr2f - last_Mer)
        last_Mer = MEr2f
        if (np.all(error_change[-4:-1]>=0)):
            break
    cf, e1f, e2f, zf, Erf, MErf = bestOrbitInnerParams(rf,sf,time,oppositions)
    return rf,sf,cf,e1f,e2f,zf,Erf,MErf
# End Q5

if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    data = pd.DataFrame(data)
    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))
