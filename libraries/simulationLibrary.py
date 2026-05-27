# !! Note!!
# in Matlab code, xh is computed in f() and placed in x(kk+1,:)
# here, xh is computed in g() and placed in y(kk,:)
# consequently, there is a difference between the Matlab and Python implementation in y_{10}
# another differences due to discretization method and sample period choice 

import numpy as np
import numpy.matlib

import pdb #pdb.set_trace()

def DefineParameters():
    # Model parameters
    p = {}
      
    # model parameters
    p['miu'] = 0.018;                  # transfer from N to S (day)
    p['landa_0'] = 0.0017;             # Basal transfer from S to R (day)
    p['landa_1'] = 0.0063;             # X-dependant transfer from S to R (day)
    p['gamma'] = 0.032;                # Pregnancy dependant transfer from S to R (day)
    p['tau'] = 0.1;                    # Milking dependant transfer from S to R (day)
    p['sigma'] = 0.0018;               # Transfer from R to A (day)
    p['phi'] = 285;                    # Maximal pregnancy length (day)
    p['alpha'] = 0.0111;               # Rate of decay of foetal growth rate
    p['omega'] = 3.5e-6;               # Initial value of foetal growth (kg)
    
    p['delta'] = 0.17;                 # Digestive contents:non-labile body mass (kg/kg)
    p['X_M'] = 0.33;                   # Mature labile:non-labile body mass (kg/kg)
    p['X_s'] = 0.20;                   # Safeguarded labile:non-labile body mass (kg/kg)
    p['psi'] = 0.58;                   # Foetus:gravid uterus mass (kg/kg)
    p['beta_X'] = 0.75;                # Fat:labile body mass (kg/kg)
    p['beta_W'] = 0.03;                # Fat:non-labile body mass (kg/kg)
    
    p['b_0'] = 1.6;                    # Balancing (day)
    p['m_0'] = 0.03;                   # Mobilization (day)
    p['k_0'] = 1.1;                    # Digestive tract removal (day)
    
    p['W_M'] = 500;                    # Mature non-labile body mass (kg)
    p['nu_x'] = 1.1;                   # Labile body mass mobilization index parameter scaling individual
    p['nu_y'] = 1.3;                   # Milk yield parameter scaling individual
    p['nu_f'] = 1;                     # Milk fat secretion parameter scaling individual
    p['nu_p'] = 1.1;                   # Milk protein secretion parameter scaling individual
    p['nu_l'] = 1;                     # Milk lactose secretion parameter scaling individual
    p['nu_h'] = 1;                     # Milk water secretion parameter scaling individual
    
    p['P_x'] = 5.7;                    # Labile body mass balance ability
    p['P_y'] = 5.0;                    # Milk secretion ability
    
    p['y_fn'] = 1.26;                  # fat scaling exponent for milk constituent secretion ability
    p['y_pn'] = 1.59;                  # protein scaling exponent for milk constituent secretion ability
    p['y_ln'] = 1.13;                  # lactose scaling exponent for milk constituent secretion ability
    p['y_hn'] = 1.15;                  # water scaling exponent for milk constituent secretion ability
    p['y_fs'] = 0.97;                  # fat scaling exponent for milk constituent secretion ability
    p['y_ps'] = 0.89;                  # protein scaling exponent for milk constituent secretion ability
    p['y_ls'] = 1.27;                  # lactose scaling exponent for milk constituent secretion ability
    p['y_hs'] = 1.21;                  # water scaling exponent for milk constituent secretion ability
    
    p['k_fn'] = 0.0035;                # fat scaling coefficient of milk constituent secretion (kg/kg)
    p['k_pn'] = 0.0030;                # protein scaling coefficient of milk constituent secretion (kg/kg)
    p['k_ln'] = 0.0037;                # lactose scaling coefficient of milk constituent secretion (kg/kg)
    p['k_hn'] = 0.0632;                # water scaling coefficient of milk constituent secretion (kg/kg)
    p['k_fs'] = 0.0032;                # fat scaling coefficient of milk constituent secretion (kg/kg)
    p['k_ps'] = 0.0027;                # protein scaling coefficient of milk constituent secretion (kg/kg)
    p['k_ls'] = 0.0049;                # lactose scaling coefficient of milk constituent secretion (kg/kg)
    p['k_hs'] = 0.0848;                # water scaling coefficient of milk constituent secretion (kg/kg)
    
    p['e_U'] = 29.299;                 # Pregnancy requirement (MJ/kg)
    p['e_M'] = 0.599;                  # Maintenance requirement (MJ/kg^0.75)
    p['e_F'] = 59.814;                 # Milk fat secretion requirement (MJ/kg)
    p['e_P'] = 32.287;                 # Milk protein secretion requirement (MJ/kg)
    p['e_L'] = 25.799;                 # Milk lactose secretion requirement (MJ/kg)
    p['e_H'] = 0.603;                  # Milk water secretion requirement (MJ/kg)
    
    p['theta_P'] = 5;                  # scaling parameter
    p['theta_Y'] = 5;                  # scaling parameter
    p['theta_AW'] = 5;                 # scaling parameter
    
    p['X_p']  = 0.16;                  # level of protection of the pregnancy function
    p['X_AW'] = 0.1;                   # level of body reserve at which energy deficit starts
    p['X_Y']  = 0.22;                  # body reserves recovery prevails over milk yield
    p['X_l']  = 0.3;                   # threshold of labile to non-labile body mass ratio
    
    p['EBW_B']   = 0.197*(p['W_M']*(1 + p['X_M']))**0.83; # empty body weight at birth
    p['W_B']     = 0.97*p['EBW_B'];                       # Non labile body mass weight at birth (kg)
    p['epsilon'] = p['omega']/p['W_B'];      
    p['Z']       = p['epsilon']*np.exp(-np.log(p['epsilon'])/(1-np.exp(-p['alpha']*p['phi'])));
   
    p['PCI']     = 83;                                    # Parturition to conception interval
    
    ## 
    # Simulation parameters
    ops       = {}
    ops['Tp'] = 14;
    ops['nr'] = 0;       # #references
    ops['ny'] = 10;      # #measurements
    ops['nx'] = 11;      # #states
    ops['nu'] = 9;       # #control signals
    ops['nd'] = 0;       # #disturbances
    
    ##
    b       = {}  
    b['s']  = 1186;
    b['q1'] = 1;
    b['q2'] = 0;
    b['q3'] = 2;
    b['c1'] = 0;
    b['c2'] = 0;    
    
    return p,ops,b


def DefineSignals(ops, u):
 
    y       = np.array(np.zeros((ops['ny'], ops['N'])));                          # measurement
    x       = np.array(np.zeros((ops['nx'], ops['N']+1)));                        # state
    
    # Average cow initial state (can be different depends on the genetics of the cow)
    x[:, 0] = [420.9162, 0, 157.7455, 89.0358, 0.0429, 0.0032, 0, 0.9504, 0.00027699, 0.00099466, 0.9476];
    
    u       = np.array(np.matlib.repmat(u, ops['N'], 1)).T;                         # control signal
    d       = np.array(np.zeros((ops['nd'], ops['N']+ops['Tp'])));                  # disturbance 
 
    return y,x,u,d


###############################################################################
###############################################################################
###############################################################################

## Initial continuous states
# xs[0]: Non-labile body mass
# xs[1]: Gravid uterus weight
# xs[2]: Labile body mass
# xs[3]: Digestive tract contents weight
# xs[4]: Growth priority
# xs[5]: Aging priority
# xs[6]: Unborn priority
# xs[7]: Newborn priority
# xs[8]: Suckling priority
# xs[9]: Reserve priority
def f1(xs,us,ds,p):
   
    
    ## feed characteristics
    # from CVB
    DM =  np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000;
    DOM = np.array([740,     704,    724,    882,    843,    711,    796,    609,    877])/1000;
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    DMC     = sum(us)/sum(us/DM);
    
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82; # metabolizable energy supply by the mobilization of 1kg X
    
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
 
    ###
    
    dGU_Ref = 0; # change rate of Gravid uterus weight
    P_Ref   = p['e_U']*(dGU_Ref); # Pregnancy energy requirement
    
    # Milk production
    Y_Ref   = 0;
    
    # energy requirements
    I_Ref   = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]))

    A_W_Ref = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref; # Energy for non-labile mass growth
    
    X_L     = xs[2]/xs[0]; # Labile to non-labile body mass ratio
    DMI_Ref = I_Ref/e_D; # Dry matter intake reference
    DMIC    = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # DMI capacity
    q       = 1/(1 + (X_L/p['X_l'])**5);
    
    prot_w  = 0.22*(1 - p['beta_W']*(xs[0]/p['W_M'])); # protein contents of body compartments W
    e_W     = (39.748*p['beta_W']*(xs[0]/p['W_M']) + 3.75*23.849*prot_w)/0.75; # metabolizable energy requirement for the storage of 1 kg W
    
    # regulating sub model
    I       = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref))); # Actual metabolizable energy intake
    delta_I = I - I_Ref; # differences between actual ME intake and reference cow (one objective)
    f       = I/I_Ref;
    Psi     = 1/(1 + f**20);
    
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)
    Theta_AW = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    Theta_X  = 1 - (Theta_AW);
    delta_BX = delta_I*Theta_X;
    delta_AX = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) +  (delta_BX - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) - C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW = Theta_AW*delta_I;
    
    A_W      = A_W_Ref + delta_AW; # Actual non labile mass growth allocation energy
    A_X      = A_X_Ref + delta_AX; # Actual non labile mass anabolism
    C_X      = C_X_Ref + delta_CX; # Actual non labile mass catabolism
    
    
    # metabolic states
    Delta_x0 = A_W/e_W;
    Delta_x1 = 0;
    Delta_x2 = A_X/e_Xa - C_X/e_Xc;
    Delta_x3 = (DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)))/ DMC - p['k_0']*(xs[3] - p['delta']*xs[0]);
    # priority states
    Delta_x4 = -((4*p['W_M']**(1/4)*(xs[0] + Delta_x0)**(3/4))**(-1))*Delta_x0;
    Delta_x5 = 0;
    Delta_x6 = 0;
    Delta_x7 = 0;
    Delta_x8 = 0;
    Delta_x9 = - Delta_x4 - Delta_x5 - Delta_x6 - Delta_x7 - Delta_x8;

    xs_new     = np.zeros(xs.shape);
    xs_new[0]  = xs[0] + Delta_x0; # Non-labile body mass (kg)
    xs_new[1]  = xs[1] + Delta_x1; # Gravid uterus mass (kg)
    xs_new[2]  = xs[2] + Delta_x2; # Labile body mass (kg)
    xs_new[3]  = xs[3] + Delta_x3; # Digestive tract contents mass (kg)
    xs_new[4]  = xs[4] + Delta_x4; # Growth priority
    xs_new[5]  = xs[5] + Delta_x5; # Aging priority
    xs_new[10] = xs[6];
    xs_new[6]  = xs[6] + Delta_x6; # Unborn priority
    xs_new[7]  = xs[7] + Delta_x7; # Newborn priority
    xs_new[8]  = xs[8] + Delta_x8; # Suckling priority
    xs_new[9]  = xs[9] + Delta_x9; # Reserves priority         
           
    return xs_new

def g1(xs,us,ds,p,b):
    
    ## feed characteristics
    DM  = np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000; # Dry matter
    NDF = np.array([494,     539,    352,    128,    119,    260,    313,    506,    135])/1000; # Neutral Detergent Fiber
    EE  = np.array([38,      36,     33,     43,     17,     20,     8,      104,    17])/1000;  # Ether Extract
    OEB = np.array([32,      19,     -41,    -50,    201,    180,    -52,    58,     -17]);      # Rumen Degradable Protein Balance
    DVE = np.array([102,     80,     52,     96,     286,    165,    102,    131,    93]);       # Digestible True Protein 
    DOM = np.array([740,     704,    724,    882,    843,   711,     796,    609,    877])/1000; # Digestible organic matter
    price=np.array([250,     222,    147,    234.1,  452,   327.1,   275.2,  273,    256.5])/1000; # feed price
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82;      # metabolizable energy supply by the mobilization of 1kg X
    
    ## mode initialization
   
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###
    
    dGU_Ref = 0; # change rate of Gravid uterus weight
    P_Ref   = p['e_U']*(dGU_Ref); # Pregnancy energy requirement
    
    # Milk production
    Y_Ref   = 0;
    
    
    # energy requirements
    I_Ref   = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
   
    X_L     = xs[2]/xs[0]; # Labile to non-labile body mass ratio
    DMI_Ref = I_Ref/e_D; # Dry matter intake reference
    DMIC    = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # DMI capacity
    q       = 1/(1 + (X_L/p['X_l'])**5);
    
    EBF     = p['beta_W']*xs[0]*xs[0]/p['W_M'] + p['beta_X']*xs[2]; # empty body fat content (kg)
    
    
    ## Outputs
    y_1 = DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)); # DMI (kg)
    y_2 = xs[0] + xs[2] + xs[1] + xs[3];                     # Body mass (kg)
    
    
    DVE_intake = y_1*us.dot(DVE)/100;                             # DVE intake
    DVE_main   = (2.75*y_2**(0.5) + 0.2*y_2**(0.6))/0.67;    # DVE for maintenance
    DVE_add    = 0;                                          # additive DVE for growth
    DVE_req    = DVE_main + DVE_add;
    
    # Body condition score (1-5)
    y_3         = (EBF/(1 - np.exp(-0.0068*(xs[0] + xs[2] - EBF)))**3.7235 - 17.9)/39.8; 
    y_4         = 0;                                            # Milk production (kg)
    expense     = y_1*(us.dot(price)/100);                           # Feed cost
    feed_margin = -expense;                                     # feed margin = milk revenue - feed cost
    y_5         = y_1*(14 - 0.375*us.dot(EE) + 0.224*us.dot(NDF));        # Enteric methane production (g)
    
    OEB_intake  = y_1*(us.dot(OEB)/100);
    FM_weigth   = y_1*sum(us/DM)/100;                             # Fresh matter intake (kg)
    
    ys = [y_1, y_2, y_3, y_4, y_5, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weigth];
           
    return ys

###############################################################################
###############################################################################
###############################################################################

def f2(xs,us,ds,p):

    ## feed characteristics
    # from CVB
    DM =  np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000;
    DOM = np.array([740,     704,    724,    882,    843,    711,    796,    609,    877])/1000;
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    DMC     = sum(us)/sum(us/DM);
    
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82; # metabolizable energy supply by the mobilization of 1kg X
    
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###  
    
    Delta_x11 = xs[6] - xs[10];
    dGU_Ref   = Delta_x11*p['W_B']/p['psi'];   # change rate of Gravid uterus weight
    P_Ref     = p['e_U']*(dGU_Ref);           # Pregnancy energy requirement
    Y_Ref     = 0;                            # Milk energy requirement
    # energy requirement
    I_Ref     = (M_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    A_W_Ref   = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref;  # Energy for non-labile mass growth
    
    X_L       = xs[2]/xs[0];                  # Labile to non-labile body mass ratio
    DMI_Ref   = I_Ref/e_D;                    # Dry matter intake reference
    DMIC      = (e_D/e_D_Ref)*I_Ref/e_D_Ref;
    q         = 1/(1 + (X_L/p['X_l'])**5);
    
    prot_w    = 0.22*(1 - p['beta_W']*(xs[0]/p['W_M']));                         # protein contents of body compartments W
    e_W       = (39.748*p['beta_W']*(xs[0]/p['W_M']) + 3.75*23.849*prot_w)/0.75; # metabolizable energy requirement for the storage of 1 kg W
    
    # regulating sub model
    I         = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref))); # Actual metabolizable energy intake
    delta_I   = I - I_Ref; # differences between actual ME intake and reference cow
    f         = I/I_Ref;
    Psi       = 1/(1 + f**20);
    
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)
    Theta_Y  = xs[8]*(1 - Psi) - (xs[8]*(1 - Psi) - Y_Ref/I_Ref*Psi)/(1 + (X_L/p['X_Y'])**p['theta_Y']);
    Theta_P  = (P_Ref/I_Ref*Psi)/(1 + (X_L/p['X_p'])**p['theta_P']);
    Theta_AW = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    Theta_M  = 0;
    Theta_X  = 1 - (Theta_Y + Theta_P + Theta_M + Theta_AW);
    delta_BX = delta_I*Theta_X;
    delta_AX = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) + (delta_BX - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) + -C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW = Theta_AW*delta_I;
    delta_P  = delta_I*Theta_P;
    
    P        = P_Ref + delta_P;     # Actual pregnancy allocation energy
    A_W      = A_W_Ref + delta_AW;  # Actual non labile mass growth allocation energy
    A_X      = A_X_Ref + delta_AX;  # Actual non labile mass anabolism
    C_X      = C_X_Ref + delta_CX;  # Actual non labile mass catabolism
    
    # metabolic states (body mass)
    Delta_x0 = A_W/e_W;
    Delta_x1 = P/p['e_U'];
    Delta_x2 = A_X/e_Xa - C_X/e_Xc;
    Delta_x3 = (DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)))/DMC - p['k_0']*(xs[3] - p['delta']*xs[0]);
    # priority states
    Delta_x4 = -((4*p['W_M']**(1/4)*(xs[0] + Delta_x0)**(3/4))**(-1))*Delta_x0;
    Delta_x5 = p['sigma']*xs[5]*(1 - xs[5]);
    Delta_x6 = p['alpha']*np.log(p['Z']/xs[6])*xs[6];
    Delta_x7 = 0;
    Delta_x8 = -(p['landa_0'] + p['landa_1']/(1 + (((xs[2] + Delta_x2)/(xs[0] + Delta_x0))/p['X_s'])**5) + p['gamma']*xs[6] + p['tau'])*xs[8];
    Delta_x9 = - Delta_x4 - Delta_x5 - Delta_x6 - Delta_x7 - Delta_x8;
    
    xs_new     = np.zeros(xs.shape);
    xs_new[0]  = xs[0] + Delta_x0; # Non-labile body mass (kg)
    xs_new[1]  = xs[1] + Delta_x1; # Gravid uterus mass (kg)
    xs_new[2]  = xs[2] + Delta_x2; # Labile body mass (kg)
    xs_new[3]  = xs[3] + Delta_x3; # Digestive tract contents mass (kg)
    xs_new[4]  = xs[4] + Delta_x4; # Growth priority
    xs_new[5]  = xs[5] + Delta_x5; # Aging priority
    xs_new[10] = xs[6];
    xs_new[6]  = xs[6] + Delta_x6; # Unborn priority
    xs_new[7]  = xs[7] + Delta_x7; # Newborn priority
    xs_new[8]  = xs[8] + Delta_x8; # Suckling priority
    xs_new[9]  = xs[9] + Delta_x9; # Reserves priority         
           
    return xs_new

def g2(xs,us,ds,p,b):

    ## feed characteristics
    DM  = np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000; # Dry matter
    NDF = np.array([494,     539,    352,    128,    119,    260,    313,    506,    135])/1000; # Neutral Detergent Fiber
    EE  = np.array([38,      36,     33,     43,     17,     20,     8,      104,    17])/1000;  # Ether Extract
    OEB = np.array([32,      19,     -41,    -50,    201,    180,    -52,    58,     -17]);      # Rumen Degradable Protein Balance
    DVE = np.array([102,     80,     52,     96,     286,    165,    102,    131,    93]);       # Digestible True Protein 
    DOM = np.array([740,     704,    724,    882,    843,   711,     796,    609,    877])/1000; # Digestible organic matter
    price=np.array([250,     222,    147,    234.1,  452,   327.1,   275.2,  273,    256.5])/1000; # feed price
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82;      # metabolizable energy supply by the mobilization of 1kg X
    
    ## mode initialization
   
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###
    
    dGU_Ref = (xs[6] - xs[10])*p['W_B']/p['psi']; # change rate of Gravid uterus weight
    P_Ref   = p['e_U']*(dGU_Ref); # Pregnancy energy requirement
    
    # energy requirement
    I_Ref   = (M_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    
    X_L     = xs[2]/xs[0]; # Labile to non-labile body mass ratio
    DMI_Ref = I_Ref/e_D; # Dry matter intake reference
    DMIC    = (e_D/e_D_Ref)*I_Ref/e_D_Ref;
    q       = 1/(1 + (X_L/p['X_l'])**5);
    EBF     = p['beta_W']*xs[0]*xs[0]/p['W_M'] + p['beta_X']*xs[2]; # empty body fat content (kg)
    
    # DVE requirement for pregnancy
    if b['c2'] > 90 and b['c2'] <= 120:
        DVE_preg = 15;
    elif b['c2'] > 120 and b['c2'] <= 150:
        DVE_preg = 30;
    elif b['c2'] > 150 and b['c2'] <= 180:
        DVE_preg = 55;
    elif b['c2'] > 180 and b['c2'] <= 210:
        DVE_preg = 100;
    elif b['c2'] > 210 and b['c2'] <= 240:
        DVE_preg = 160;
    elif b['c2'] > 240:
        DVE_preg = 255;
    else: 
        DVE_preg = 0;
    
    # DVE for growth
    if b['q3'] == 1:
        DVE_add = 64;
    elif b['q3'] == 2:
        DVE_add = 37;
    elif b['q3'] == 3 or b['q3'] == 4:
        DVE_add = 22;
    else: 
        DVE_add = 0;
    
    ## Outputs
    y_1         = DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)); # DMI (kg)
    y_2         = xs[0] + xs[2] + xs[1] + xs[3];                   # Body mass (kg)
    
    DVE_intake  = y_1*us.dot(DVE)/100;                             # DVE intake
    DVE_main    = (2.75*y_2**(0.5) + 0.2*y_2**(0.6))/0.67;         # DVE for maintenance
    DVE_req     = DVE_main + DVE_preg + DVE_add;
    
    # Body condition score (1-5)
    y_3         = (EBF/(1 - np.exp(-0.0068*(xs[0] + xs[2] - EBF)))**3.7235 - 17.9)/39.8; 
    y_4         = 0;                                               # milk production
    expense     = y_1*(us.dot(price)/100);
    feed_margin = -expense;
    y_5         = y_1*(14 - 0.375*us.dot(EE) + 0.224*us.dot(NDF)); # Enteric methane production (g)
    OEB_intake  = y_1*(us.dot(OEB)/100);
    FM_weigth   = y_1*sum(us/DM)/100;                              # Fresh matter intake (kg)
    
    ys = [y_1, y_2, y_3, y_4, y_5, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weigth];
                  
    return ys

###############################################################################
###############################################################################
###############################################################################
def f3(xs,us,ds,p):
    
    ## feed characteristics
    # from CVB
    DM =  np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000;
    DOM = np.array([740,     704,    724,    882,    843,    711,    796,    609,    877])/1000;
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    DMC     = sum(us)/sum(us/DM);
    
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82; # metabolizable energy supply by the mobilization of 1kg X
    
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
 
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###          
    
    dGU_Ref = 0;               # change rate of Gravid uterus weight
    P_Ref   = p['e_U']*(dGU_Ref); # Pregnancy energy requirement
    
    # Milk production
    # daily yield of milk constituent: Fat, protein, water, lactose
    MY_F_Ref = (1 - xs[4])**p['P_y']*(p['k_fn']*xs[7]**p['y_fn'] + p['k_fs']*xs[8]**p['y_fs'])*p['W_M']*p['nu_y']*p['nu_f'];
    MY_P_Ref = (1 - xs[4])**p['P_y']*(p['k_pn']*xs[7]**p['y_pn'] + p['k_ps']*xs[8]**p['y_ps'])*p['W_M']*p['nu_y']*p['nu_p'];
    MY_H_Ref = (1 - xs[4])**p['P_y']*(p['k_hn']*xs[7]**p['y_hn'] + p['k_hs']*xs[8]**p['y_hs'])*p['W_M']*p['nu_y']*p['nu_h'];
    MY_L_Ref = (1 - xs[4])**p['P_y']*(p['k_ln']*xs[7]**p['y_ln'] + p['k_ls']*xs[8]**p['y_ls'])*p['W_M']*p['nu_y']*p['nu_l'];
    
    # Milk energy requirement
    Y_Ref    = p['e_F']*MY_F_Ref + p['e_P']*MY_P_Ref + p['e_H']*MY_H_Ref + p['e_L']*MY_L_Ref;
    
    # energy requirement
    I_Ref    = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    A_W_Ref  = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref;              # Energy for non-labile mass growth
    
    X_L      = xs[2]/xs[0];                 # Labile to non-labile body mass ratio
    DMI_Ref  = I_Ref/e_D;                   # Dry matter intake reference
    DMIC     = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # Dry matter capacity
    q        = 1/(1 + (X_L/p['X_l'])**5);
    
    prot_w   = 0.22*(1 - p['beta_W']*(xs[0]/p['W_M']));                         # protein contents of body compartments W
    e_W      = (39.748*p['beta_W']*(xs[0]/p['W_M']) + 3.75*23.849*prot_w)/0.75; # metabolizable energy requirement for the storage of 1 kg W
    
    # regulating sub model (allocation in Puillet)
    I        = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref))); # Actual metabolizable energy intake
    delta_I  = I - I_Ref; # differences between actual ME intake and reference cow
    f        = I/I_Ref;
    Psi      = 1/(1 + f**20);
    
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)
    Theta_Y   = xs[8]*(1 - Psi) - (xs[8]*(1 - Psi) - Y_Ref/I_Ref*Psi)/(1 + (X_L/p['X_Y'])**p['theta_Y']);
    Theta_P   = (P_Ref/I_Ref*Psi)/(1 + (X_L/p['X_p'])**p['theta_P']);
    Theta_AW  = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    Theta_M   = 0;
    Theta_X   = 1 - (Theta_Y + Theta_P + Theta_M + Theta_AW);
    delta_BX  = delta_I*Theta_X;
    delta_AX  = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) + (delta_BX - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX  = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) + -C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW  = Theta_AW*delta_I;
    
    A_W       = A_W_Ref + delta_AW; # Actual non labile mass growth allocation energy
    A_X       = A_X_Ref + delta_AX; # Actual non labile mass anabolism
    C_X       = C_X_Ref + delta_CX; # Actual non labile mass catabolism
    
    # Metabolic states (Body mass)
    Delta_x0 = A_W/e_W;
    Delta_x1 = 0;
    Delta_x2 = A_X/e_Xa - C_X/e_Xc;
    Delta_x3 = (DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)))/DMC - p['k_0']*(xs[3] - p['delta']*xs[0]);
    # Priority states
    Delta_x4 = -((4*p['W_M']**(1/4)*(xs[0] + Delta_x0)**(3/4))**(-1))*Delta_x0;
    Delta_x5 = p['sigma']*xs[5]*(1 - xs[5]);
    Delta_x6 = 0;
    Delta_x7 = -p['miu']*xs[7];
    Delta_x8 = p['miu']*xs[7] - (p['landa_0'] + p['landa_1']/(1 + (((xs[2] + Delta_x2)/(xs[0] + Delta_x0))/p['X_s'])**5) + p['gamma']*xs[6])*xs[8];
    Delta_x9 = - Delta_x4 - Delta_x5 - Delta_x6 - Delta_x7 - Delta_x8;
    
    xs_new     = np.zeros(xs.shape);
    xs_new[0]  = xs[0] + Delta_x0; # Non-labile body mass (kg)
    xs_new[1]  = xs[1] + Delta_x1; # Gravid uterus mass (kg)
    xs_new[2]  = xs[2] + Delta_x2; # Labile body mass (kg)
    xs_new[3]  = xs[3] + Delta_x3; # Digestive tract contents mass (kg)
    xs_new[4]  = xs[4] + Delta_x4; # Growth priority
    xs_new[5]  = xs[5] + Delta_x5; # Aging priority
    xs_new[10] = xs[6];
    xs_new[6]  = xs[6] + Delta_x6; # Unborn priority
    xs_new[7]  = xs[7] + Delta_x7; # Newborn priority
    xs_new[8]  = xs[8] + Delta_x8; # Suckling priority
    xs_new[9]  = xs[9] + Delta_x9; # Reserves priority      
     
    return xs_new





def g3(xs,us,ds,p,b):

    ## feed characteristics
    DM  = np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000; # Dry matter
    NDF = np.array([494,     539,    352,    128,    119,    260,    313,    506,    135])/1000; # Neutral Detergent Fiber
    EE  = np.array([38,      36,     33,     43,     17,     20,     8,      104,    17])/1000;  # Ether Extract
    OEB = np.array([32,      19,     -41,    -50,    201,    180,    -52,    58,     -17]);      # Rumen Degradable Protein Balance
    DVE = np.array([102,     80,     52,     96,     286,    165,    102,    131,    93]);       # Digestible True Protein 
    DOM = np.array([740,     704,    724,    882,    843,   711,     796,    609,    877])/1000; # Digestible organic matter
    price=np.array([250,     222,    147,    234.1,  452,   327.1,   275.2,  273,    256.5])/1000; # feed price
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82;      # metabolizable energy supply by the mobilization of 1kg X
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    ## mode initialization
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###        

    dGU_Ref = 0; # change rate of Gravid uterus weight
    P_Ref   = p['e_U']*(dGU_Ref); # Pregnancy energy requirement      

    # Milk production
    # daily yield of milk constituent: Fat, protein, water, lactose
    MY_F_Ref = (1 - xs[4])**p['P_y']*(p['k_fn']*xs[7]**p['y_fn'] + p['k_fs']*xs[8]**p['y_fs'])*p['W_M']*p['nu_y']*p['nu_f'];
    MY_P_Ref = (1 - xs[4])**p['P_y']*(p['k_pn']*xs[7]**p['y_pn'] + p['k_ps']*xs[8]**p['y_ps'])*p['W_M']*p['nu_y']*p['nu_p'];
    MY_H_Ref = (1 - xs[4])**p['P_y']*(p['k_hn']*xs[7]**p['y_hn'] + p['k_hs']*xs[8]**p['y_hs'])*p['W_M']*p['nu_y']*p['nu_h'];
    MY_L_Ref = (1 - xs[4])**p['P_y']*(p['k_ln']*xs[7]**p['y_ln'] + p['k_ls']*xs[8]**p['y_ls'])*p['W_M']*p['nu_y']*p['nu_l'];
    
    # Milk energy requirement
    Y_Ref    = p['e_F']*MY_F_Ref + p['e_P']*MY_P_Ref + p['e_H']*MY_H_Ref + p['e_L']*MY_L_Ref;
    Y_F_Ref  = p['e_F']*MY_F_Ref;   # energy requirement for fat yield
    Y_P_Ref  = p['e_P']*MY_P_Ref;   # energy requirement for protein yield
    Y_H_Ref  = p['e_H']*MY_H_Ref;   # energy requirement for water yield
    Y_L_Ref  = p['e_L']*MY_L_Ref;   # energy requirement for lactose yield

    # energy requirement
    I_Ref    = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    A_W_Ref  = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref;              # Energy for non-labile mass growth
    
    X_L      = xs[2]/xs[0];                 # Labile to non-labile body mass ratio
    DMI_Ref  = I_Ref/e_D;                   # Dry matter intake reference
    DMIC     = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # Dry matter capacity
    q        = 1/(1 + (X_L/p['X_l'])**5);
    EBF      = p['beta_W']*xs[0]*xs[0]/p['W_M'] + p['beta_X']*xs[2]; # empty body fat content (kg)

    # regulating sub model (allocation in Puillet)
    I       = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)));  # Actual metabolizable energy intake
    delta_I = I - I_Ref;                                                                                                                  # differences between actual ME intake and reference cow
    f       = I/I_Ref;
    Psi     = 1/(1 + f**20);
    
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)
    Theta_Y   = xs[8]*(1 - Psi) - (xs[8]*(1 - Psi) - Y_Ref/I_Ref*Psi)/(1 + (X_L/p['X_Y'])**p['theta_Y']);
    Theta_P   = (P_Ref/I_Ref*Psi)/(1 + (X_L/p['X_p'])**p['theta_P']);
    Theta_AW  = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    Theta_M   = 0;
    Theta_X   = 1 - (Theta_Y + Theta_P + Theta_M + Theta_AW);
    delta_BX  = delta_I*Theta_X;
    delta_AX  = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) + (delta_BX  - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX  = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) - C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW  = Theta_AW*delta_I;
    delta_M   = 0;
    delta_P   = delta_I*Theta_P;
    delta_Y_D = (delta_I - delta_M - delta_P - delta_AW - delta_AX);
    delta_Y_X = delta_CX;
    
    U_D_F = Y_F_Ref/Y_Ref;
    U_D_P = Y_P_Ref/Y_Ref;
    U_D_H = Y_H_Ref/Y_Ref;
    U_D_L = Y_L_Ref/Y_Ref;
    Y_P_e = Y_P_Ref + U_D_P*delta_Y_D;
    Y_H   = Y_H_Ref + U_D_H*delta_Y_D;
    Y_L   = Y_L_Ref + U_D_L*delta_Y_D;
    
    # DVE for growth
    if b['q3'] == 1:
        DVE_add = 64;
    elif b['q3'] == 2:
        DVE_add = 37;
    elif b['q3'] == 3 or b['q3'] == 4:
        DVE_add = 22;
    else:
        DVE_add = 0;

    
    ## Outputs
    y_1 = DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)); # DMI (kg)
    y_2 = xs[0] + xs[2] + xs[1] + xs[3];                                                                                       # Body mass (kg)
    
    DVE_intake = y_1*us.dot(DVE)/100;
    DVE_main   = (2.75*y_2**(0.5) + 0.2*y_2**(0.6))/0.67; # DVE for maintenance
    DVE_nes    = DVE_main + DVE_add;
    DVE_ex     = DVE_intake - DVE_nes;                    # DVE extera for milk production
    
    Y_P_p      = (-1.396 + np.sqrt(1.396**2 + 0.00078*DVE_ex))/0.00039;
    MY_P       = min(Y_P_e/p['e_P'], Y_P_p/1000);
    DVE_milk   = 1.396*(1000*MY_P) + 0.000195*(1000*MY_P)**2;
    
    DVE_req    = DVE_main + DVE_add + DVE_milk;
    
    y_3        = (EBF/(1 - np.exp(-0.0068*(xs[0] + xs[2] - EBF)))**3.7235 - 17.9)/39.8; # Body condition score (1-5)
    
    y_4        = ((Y_F_Ref + U_D_F*delta_Y_D)/p['e_F'] + delta_Y_X/(0.75*p['e_F'])) + MY_P + Y_H/p['e_H'] + Y_L/p['e_L']; # Milk production (kg)
    MC_F       = ((Y_F_Ref + U_D_F*delta_Y_D)/p['e_F'] + delta_Y_X/(0.75*p['e_F']))/y_4;
    MC_P       = MY_P/y_4;
    
    ECM         = y_4*(0.383*MC_F*100 + 0.242*MC_P*100 + 0.7832) / 3.140;
    M_price     = 54*ECM/100;                 # milk revenue
    expense     = y_1*(us.dot(price)/100);      # feed cost
    feed_margin = M_price - expense;        # feed margin
    
    y_5         = y_1*(14 - 0.375*us.dot(EE) + 0.224*us.dot(NDF)); # Enteric methane production (g)
    
    OEB_intake  = y_1*(us.dot(OEB)/100);
    FM_weigth   = y_1*sum(us/DM)/100;    # fresh weight
    
    ys = [y_1, y_2, y_3, y_4, y_5, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weigth];
           
    return ys

###############################################################################
###############################################################################
###############################################################################

def f4(xs,us,ds,p):
         
    ## feed characteristics
    # from CVB
    DM =  np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000;
    DOM = np.array([740,     704,    724,    882,    843,    711,    796,    609,    877])/1000;
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    DMC     = sum(us)/sum(us/DM);
    
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82; # metabolizable energy supply by the mobilization of 1kg X
    
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
 
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###

    Delta_x11 = xs[6] - xs[10];
    dGU_Ref   = Delta_x11*p['W_B']/p['psi']; # change rate of Gravid uterus weight
    P_Ref     = p['e_U']*(dGU_Ref);       # Pregnancy energy requirement
        
    # Milk production
    # daily yield of milk constituent: Fat, protein, water, lactose
    MY_F_Ref = (1 - xs[4])**p['P_y']*(p['k_fn']*xs[7]**p['y_fn'] + p['k_fs']*xs[8]**p['y_fs'])*p['W_M']*p['nu_y']*p['nu_f'];
    MY_P_Ref = (1 - xs[4])**p['P_y']*(p['k_pn']*xs[7]**p['y_pn'] + p['k_ps']*xs[8]**p['y_ps'])*p['W_M']*p['nu_y']*p['nu_p'];
    MY_H_Ref = (1 - xs[4])**p['P_y']*(p['k_hn']*xs[7]**p['y_hn'] + p['k_hs']*xs[8]**p['y_hs'])*p['W_M']*p['nu_y']*p['nu_h'];
    MY_L_Ref = (1 - xs[4])**p['P_y']*(p['k_ln']*xs[7]**p['y_ln'] + p['k_ls']*xs[8]**p['y_ls'])*p['W_M']*p['nu_y']*p['nu_l'];
    
    
    # Milk energy requirement
    Y_Ref    = p['e_F']*MY_F_Ref + p['e_P']*MY_P_Ref + p['e_H']*MY_H_Ref + p['e_L']*MY_L_Ref;
    
    # energy requirements
    I_Ref    = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    A_W_Ref  = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref;              # Energy for non-labile mass growth
    
    X_L      = xs[2]/xs[0];                 # Labile to non-labile body mass ratio
    DMI_Ref  = I_Ref/e_D;                   # Dry matter intake reference
    DMIC     = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # Dry matter capacity
    q        = 1/(1 + (X_L/p['X_l'])**5);
    
    prot_w   = 0.22*(1 - p['beta_W']*(xs[0]/p['W_M']));                         # protein contents of body compartments W
    e_W      = (39.748*p['beta_W']*(xs[0]/p['W_M']) + 3.75*23.849*prot_w)/0.75; # metabolizable energy requirement for the storage of 1 kg W
    
    prot_w   = 0.22*(1 - p['beta_W']*(xs[0]/p['W_M']));                         # protein contents of body compartments W
    e_W      = (39.748*p['beta_W']*(xs[0]/p['W_M']) + 3.75*23.849*prot_w)/0.75; # metabolizable energy requirement for the storage of 1 kg W    
    
    # regulating sub model (allocation in Puillet)

    I        = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref))); # Actual metabolizable energy intake
    delta_I  = I - I_Ref; # differences between actual ME intake and reference cow
    f        = I/I_Ref;
    Psi      = 1/(1 + f**20);
    
    
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)   
    Theta_Y   = xs[8]*(1 - Psi) - (xs[8]*(1 - Psi) - Y_Ref/I_Ref*Psi)/(1 + (X_L/p['X_Y'])**p['theta_Y']);
    Theta_P   = (P_Ref/I_Ref*Psi)/(1 + (X_L/p['X_p'])**p['theta_P']);
    Theta_AW  = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    delta_BX  = delta_I*(1 - (Theta_Y + Theta_P + Theta_AW));
    delta_AX  = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) + (delta_BX - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX  = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) - C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW  = Theta_AW*delta_I;
    delta_P   = delta_I*Theta_P;
    P         = P_Ref + delta_P;    # Actual pregnancy allocation energy
    A_W       = A_W_Ref + delta_AW; # Actual non labile mass growth allocation energy
    A_X       = A_X_Ref + delta_AX; # Actual non labile mass anabolism
    C_X       = C_X_Ref + delta_CX; # Actual non labile mass catabolism
       
    # Metabolic states (body mass)
    Delta_x0 = A_W/e_W;
    Delta_x1 = P/p['e_U'];
    Delta_x2 = A_X/e_Xa - C_X/e_Xc;
    Delta_x3 = (DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)))/DMC - p['k_0']*(xs[3] - p['delta']*xs[0]);
    # Priority states
    Delta_x4 = -((4*p['W_M']**(1/4)*(xs[0] + Delta_x1)**(3/4))**(-1))*Delta_x0;
    Delta_x5 = p['sigma']*xs[5]*(1 - xs[5]);
    Delta_x6 = p['alpha']*np.log(p['Z']/xs[6])*xs[6];
    Delta_x7 = - p['miu']*xs[7];
    Delta_x8 = p['miu']*xs[7] - (p['landa_0'] + p['landa_1']/(1 + (((xs[2] + Delta_x2)/(xs[0] + Delta_x0))/p['X_s'])**5) + p['gamma']*xs[6])*xs[8];
    Delta_x9 = - Delta_x4 - Delta_x5 - Delta_x6 - Delta_x7 - Delta_x8;
    
    xs_new     = np.zeros(xs.shape);
    xs_new[0]  = xs[0] + Delta_x0;   # Non-labile body mass (kg)
    xs_new[1]  = xs[1] + Delta_x1;   # Gravid uterus mass (kg)
    xs_new[2]  = xs[2] + Delta_x2;   # Labile body mass (kg)
    xs_new[3]  = xs[3] + Delta_x3;   # Digestive tract contents mass (kg)
    xs_new[4]  = xs[4] + Delta_x4;   # Growth priority
    xs_new[5]  = xs[5] + Delta_x5;   # Aging priority
    xs_new[10] = xs[6];
    xs_new[6]  = xs[6] + Delta_x6;   # Unborn priority
    xs_new[7]  = xs[7] + Delta_x7;   # Newborn priority
    xs_new[8]  = xs[8] + Delta_x8;   # Suckling priority
    xs_new[9]  = xs[9] + Delta_x9; # Reserves priority

    return xs_new

def g4(xs,us,ds,p,b):
        
    ## feed characteristics
    DM  = np.array([165,     450,    398,    863,    877,    897,    912,    265,    867])/1000; # Dry matter
    NDF = np.array([494,     539,    352,    128,    119,    260,    313,    506,    135])/1000; # Neutral Detergent Fiber
    EE  = np.array([38,      36,     33,     43,     17,     20,     8,      104,    17])/1000;  # Ether Extract
    OEB = np.array([32,      19,     -41,    -50,    201,    180,    -52,    58,     -17]);      # Rumen Degradable Protein Balance
    DVE = np.array([102,     80,     52,     96,     286,    165,    102,    131,    93]);       # Digestible True Protein 
    DOM = np.array([740,     704,    724,    882,    843,   711,     796,    609,    877])/1000; # Digestible organic matter
    price=np.array([250,     222,    147,    234.1,  452,   327.1,   275.2,  273,    256.5])/1000; # feed price
    
    # metabolizable energy intake
    den = 0;
    num = 0;
    for h in range(len(us)):
        num = num + us[h]*DOM[h];
        den = den + us[h];

    e_D     = 4.1855*(4.29*num/den - 0.31);     # Dietary ME content (DM basis) (MJ/kg)
    e_D_Ref = 11.297;                           # Reference Dietary content
    
    e_Xa    = (39.748*p['beta_X'] + 3.75*23.849*0.22*(1 - p['beta_X']))/0.75; # metabolizable energy requirement for the storage of 1 kg X
    e_Xc    = (39.748*p['beta_X'] + 23.849*0.22*(1 - p['beta_X']))*0.82;      # metabolizable energy supply by the mobilization of 1kg X
    x_5     = 1 - (p['W_B']/p['W_M'])**(1/4);
    
    ## mode initialization
    M_Ref   = p['e_M']*(xs[0] + xs[2] + xs[1] + xs[3])**0.75; # Maintenance energy requirement
    # Change in Labile body mass
    dX_Ref  = (1 - xs[4])**p['P_x']*(p['b_0']*(1 - (xs[2]/p['W_M'])/p['X_M'])*xs[9] - p['m_0']*xs[7]*xs[2]*p['nu_x']);
    
    # dx = reference balance of labile body mass - reference mobilization of labile body mass
    A_X_Ref = e_Xa*dX_Ref*smooth(dX_Ref);
    C_X_Ref = -e_Xc*dX_Ref*smooth(-dX_Ref);
    
    ###
    
    Delta_x10 = xs[6] - xs[10];
    dGU_Ref   = Delta_x10*p['W_B']/p['psi'];   # change rate of Gravid uterus weight
    P_Ref     = p['e_U']*(dGU_Ref);         # Pregnancy energy requirement
      
    # Milk production
    # daily yield of milk constituent: Fat, protein, water, lactose
    MY_F_Ref = (1 - xs[4])**p['P_y']*(p['k_fn']*xs[7]**p['y_fn'] + p['k_fs']*xs[8]**p['y_fs'])*p['W_M']*p['nu_y']*p['nu_f'];
    MY_P_Ref = (1 - xs[4])**p['P_y']*(p['k_pn']*xs[7]**p['y_pn'] + p['k_ps']*xs[8]**p['y_ps'])*p['W_M']*p['nu_y']*p['nu_p'];
    MY_H_Ref = (1 - xs[4])**p['P_y']*(p['k_hn']*xs[7]**p['y_hn'] + p['k_hs']*xs[8]**p['y_hs'])*p['W_M']*p['nu_y']*p['nu_h'];
    MY_L_Ref = (1 - xs[4])**p['P_y']*(p['k_ln']*xs[7]**p['y_ln'] + p['k_ls']*xs[8]**p['y_ls'])*p['W_M']*p['nu_y']*p['nu_l'];
    
    
    # Milk energy requirement
    Y_Ref    = p['e_F']*MY_F_Ref + p['e_P']*MY_P_Ref + p['e_H']*MY_H_Ref + p['e_L']*MY_L_Ref;
    Y_F_Ref  = p['e_F']*MY_F_Ref;   # energy requirement for fat yield
    Y_P_Ref  = p['e_P']*MY_P_Ref;   # energy requirement for protein yield
    Y_H_Ref  = p['e_H']*MY_H_Ref;   # energy requirement for water yield
    Y_L_Ref  = p['e_L']*MY_L_Ref;   # energy requirement for lactose yield
    
    # energy requirement
    I_Ref    = (M_Ref + Y_Ref + P_Ref + A_X_Ref - C_X_Ref)/(1 - xs[4]*(xs[4] + xs[5] + xs[9]));
    A_W_Ref  = xs[4]*(xs[4] + xs[5] + xs[9])*I_Ref;              # Energy for non-labile mass growth
    
    X_L      = xs[2]/xs[0];                 # Labile to non-labile body mass ratio
    DMI_Ref  = I_Ref/e_D;                   # Dry matter intake reference
    DMIC     = (e_D/e_D_Ref)*I_Ref/e_D_Ref; # Dry matter capacity
    q        = 1/(1 + (X_L/p['X_l'])**5);
    EBF      = p['beta_W']*xs[0]*xs[0]/p['W_M'] + p['beta_X']*xs[2]; # empty body fat content (kg)
    
    # regulating sub model
    I       = e_D*(DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)));  # Actual metabolizable energy intake
    delta_I = I - I_Ref;                                                                                                                  # differences between actual ME intake and reference cow
    f       = I/I_Ref;
    Psi     = 1/(1 + f**20);
   
    # Partitioning function for energy flows (Y: milk, P: Pregnancy, AW:
    # Growth, M: maintenace, CX & AX: body fat)
    Theta_Y   = xs[8]*(1 - Psi) - (xs[8]*(1 - Psi) - Y_Ref/I_Ref*Psi)/(1 + (X_L/p['X_Y'])**p['theta_Y']);
    Theta_P   = (P_Ref/I_Ref*Psi)/(1 + (X_L/p['X_p'])**p['theta_P']);
    Theta_AW  = xs[4]*(1 - Psi) - (xs[4]*(1 - Psi) - xs[4]/x_5*(1 - Psi) - A_W_Ref/I_Ref*Psi)/(1 + (X_L/p['X_AW'])**p['theta_AW']);
    
    delta_BX = delta_I*(1 - (Theta_Y + Theta_P + Theta_AW));
    delta_AX  = -A_X_Ref*smooth(-A_X_Ref - delta_BX + C_X_Ref) + (delta_BX  - C_X_Ref)*smooth(delta_BX - C_X_Ref + A_X_Ref);
    delta_CX  = (-delta_BX - A_X_Ref)*smooth(-delta_BX - A_X_Ref + C_X_Ref) - C_X_Ref*smooth(-C_X_Ref + delta_BX + A_X_Ref);
    delta_AW  = Theta_AW*delta_I;
    delta_M   = 0;
    delta_P   = delta_I*Theta_P;
    delta_Y_D = (delta_I - delta_M - delta_P - delta_AW - delta_AX);
    delta_Y_X = delta_CX;
    
    U_D_F = Y_F_Ref/Y_Ref;
    U_D_P = Y_P_Ref/Y_Ref;
    U_D_H = Y_H_Ref/Y_Ref;
    U_D_L = Y_L_Ref/Y_Ref;
    Y_P_e = Y_P_Ref + U_D_P*delta_Y_D;
    Y_H   = Y_H_Ref + U_D_H*delta_Y_D;
    Y_L   = Y_L_Ref + U_D_L*delta_Y_D;
    
 
    # DVE for pregnancy
    if b['c2'] > 90 and b['c2'] <= 120:
        DVE_preg = 15;
    elif b['c2'] > 120 and b['c2'] <= 150:
        DVE_preg = 30;
    elif b['c2'] > 150 and b['c2'] <= 180:
        DVE_preg = 55;
    elif b['c2'] > 180 and b['c2'] <= 210:
        DVE_preg = 100;
    elif b['c2'] > 210 and b['c2'] <= 240:
        DVE_preg = 160;
    elif b['c2'] > 240:
        DVE_preg = 255;
    else:
        DVE_preg = 0;
    
    # DVE for growth
    if b['q3'] == 1:
        DVE_add = 64;
    elif b['q3'] == 2:
        DVE_add = 37;
    elif b['q3'] == 3 or b['q3'] == 4:
        DVE_add = 22;
    else:
        DVE_add = 0;

    ## Outputs
    y_1 = DMIC*smooth(q*DMIC + (1 - q)*DMI_Ref - DMIC) + (q*DMIC + (1 - q)*DMI_Ref)*smooth(DMIC - (q*DMIC + (1 - q)*DMI_Ref)); # DMI (kg)
    y_2 = xs[0] + xs[2] + xs[1] + xs[3];                                                                                       # Body mass (kg)
    
    DVE_intake = y_1*us.dot(DVE)/100;
    DVE_main   = (2.75*y_2**(0.5) + 0.2*y_2**(0.6))/0.67; # DVE for maintenance
    DVE_nes    = DVE_main + DVE_preg + DVE_add;
    DVE_ex     = DVE_intake - DVE_nes;                    # DVE extera for milk production
    
    Y_P_p      = (-1.396 + np.sqrt(1.396**2 + 0.00078*DVE_ex))/0.00039;
    MY_P       = min(Y_P_e/p['e_P'], Y_P_p/1000);
    DVE_milk   = 1.396*(1000*MY_P) + 0.000195*(1000*MY_P)**2;
    
    DVE_req    = DVE_main + DVE_preg + DVE_add + DVE_milk;
    
    y_3        = (EBF/(1 - np.exp(-0.0068*(xs[0] + xs[2] - EBF)))**3.7235 - 17.9)/39.8; # Body condition score (1-5)
    
    y_4        = ((Y_F_Ref + U_D_F*delta_Y_D)/p['e_F'] + delta_Y_X/(0.75*p['e_F'])) + MY_P + Y_H/p['e_H'] + Y_L/p['e_L']; # Milk production (kg)
    MC_F       = ((Y_F_Ref + U_D_F*delta_Y_D)/p['e_F'] + delta_Y_X/(0.75*p['e_F']))/y_4;
    MC_P       = MY_P/y_4;
    
    ECM         = y_4*(0.383*MC_F*100 + 0.242*MC_P*100 + 0.7832) / 3.140;
    M_price     = 54*ECM/100;                 # milk revenue
    expense     = y_1*(us.dot(price)/100);      # feed cost
    feed_margin = M_price - expense;        # feed margin
    
    y_5         = y_1*(14 - 0.375*us.dot(EE) + 0.224*us.dot(NDF)); # Enteric methane production (g)
    
    OEB_intake  = y_1*(us.dot(OEB)/100);
    FM_weigth   = y_1*sum(us/DM)/100;    # fresh weight
    
    ys = [y_1, y_2, y_3, y_4, y_5, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weigth];
        
    return ys

###############################################################################
###############################################################################
###############################################################################


def smooth(x):   
    y = (np.tanh(200*x)+1)/2;   
    return y