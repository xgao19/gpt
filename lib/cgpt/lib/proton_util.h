
static const int epsilon[6][3] = {{0,1,2},{1,2,0},{2,0,1},{0,2,1},{2,1,0},{1,0,2}};
static const Real epsilon_sgn[6] = {1.,1.,1.,-1.,-1.,-1.};



template <class mobj>
vComplexD Proton2ptSite(const mobj &D, const int polarization){

    Gamma g4(Gamma::Algebra::GammaT);
    Gamma g5(Gamma::Algebra::Gamma5);
    Gamma g2(Gamma::Algebra::GammaY);

    auto G = Complex(0,1) * g2 * g4 * g5;

    if (polarization==0)
    {
        //polatization projector
        // Notes: P_+,Sz+, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 - g1 * g2);   
    }
    else if (polarization==1)
    {
        //polatization projector
        // Notes: P_+,Sx+, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 - g2 * g3);  
    }
    else
    {
        //polatization projector
        // Notes: P_+,Sx-, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 + g2 * g3);  
    }

    auto PD = P * D; //paritiy projected propagator
    auto GDG = G * D * G;   //propagator sandwiched between C * gamma5 matrices

    for (int ie_f=0; ie_f < 6 ; ie_f++){
        int a_f = epsilon[ie_f][0]; //a
        int b_f = epsilon[ie_f][1]; //b
        int c_f = epsilon[ie_f][2]; //c
    for (int ie_i=0; ie_i < 6 ; ie_i++){
        int a_i = epsilon[ie_i][0]; //a'
        int b_i = epsilon[ie_i][1]; //b'
        int c_i = epsilon[ie_i][2]; //c'

        Real ee = epsilon_sgn[ie_f] * epsilon_sgn[ie_i];

        for (int rho=0; rho<Ns; rho++){
            auto PD_rr_cc = PD()(rho,rho)(c_f,c_i);
            for (int alpha_f=0; alpha_f<Ns; alpha_f++){
            for (int beta_i=0; beta_i<Ns; beta_i++){
                result()()() += ee  * PD_rr_cc
                                    * D    ()(alpha_f,beta_i)(a_f,a_i)
                                    * GDG    ()(alpha_f,beta_i)(b_f,b_i);
            }}
        }



        for (int rho=0; rho<Ns; rho++){
        for (int alpha_f=0; alpha_f<Ns; alpha_f++){
            auto D_ar_ac = D()(alpha_f,rho)(a_f,c_i);
            for (int beta_i=0; beta_i<Ns; beta_i++){
                result()()() -= ee  * D_ar_ac
                                    * PD    ()(rho,beta_i)(c_f,a_i)
                                    * GDG        ()(alpha_f,beta_i)(b_f,b_i);
            }
        }}

    }}

    return result;

}

template <class mobj, class robj>
void ProtonSeqSrcSite(const mobj &F, robj &seq_src, int polarization, int flavor){

    // flavor == 1 : up
    // flavor == 2 : down
    // flavor == 0 : up - down

    Gamma g1(Gamma::Algebra::GammaX);
    Gamma g2(Gamma::Algebra::GammaY);
    Gamma g3(Gamma::Algebra::GammaZ);
    Gamma g4(Gamma::Algebra::GammaT);
    Gamma g5(Gamma::Algebra::Gamma5);    

    // projecting onto the quantum numbers of the proton
    // convention as in Gattringer-Lang = i g2 g4 g5
    auto G = Complex(0,1) * g2 * g4 * g5;

    if (polarization==0)
    {
        //polatization projector
        // Notes: P_+,Sz+, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 - g1 * g2);   
    }
    else if (polarization==1)
    {
        //polatization projector
        // Notes: P_+,Sx+, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 - g2 * g3);  
    }
    else
    {
        //polatization projector
        // Notes: P_+,Sx-, QLUA: tpol_posSzplus
        auto P = 0.5 * (1 + g4) * (1 + g2 * g3);  
    }
    
    
    

    auto PF = P * F; // polarization projected propagator

    auto GFG = G * F * G;       //propagator sandwiched between C * gamma5 matrices

    auto GF = G * F;
    auto PFG = PF * G;

    for (int ie_f=0; ie_f < 6 ; ie_f++){
        int a_f = epsilon[ie_f][0]; //a
        int b_f = epsilon[ie_f][1]; //b
        int c_f = epsilon[ie_f][2]; //c
    for (int ie_i=0; ie_i < 6 ; ie_i++){
        int a_i = epsilon[ie_i][0]; //a'
        int b_i = epsilon[ie_i][1]; //b'
        int c_i = epsilon[ie_i][2]; //c'

        Real ee = epsilon_sgn[ie_f] * epsilon_sgn[ie_i];

        for (int alpha_f=0; alpha_f<Ns; alpha_f++){
        for (int alpha_i=0; alpha_i<Ns; alpha_i++){

            
            for (int gamma_i=0; gamma_i<Ns; gamma_i++){
                if (flavor==0){
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  *
                                GFG()(alpha_f, gamma_i)(b_f,b_i) * PF()(alpha_i, gamma_i)(c_f,c_i);
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  *
                                GF()(alpha_f, gamma_i)(b_f,c_i) * PFG()(gamma_i, alpha_i)(c_f,b_i)
                }

                if (flavor==1){
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  *
                                GFG()(alpha_f, gamma_i)(b_f,b_i) * PF()(alpha_i, gamma_i)(c_f,c_i);
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  * 
                                GFG()(alpha_f,alpha_i)(b_f,b_i) * PF()(gamma_i, gamma_i)(c_f,c_i);
                }

                if (flavor==2){
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  * 
                                GFG()(alpha_f,alpha_i)(b_f,b_i) * PF()(gamma_i, gamma_i)(c_f,c_i);
                    seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  *
                                GF()(alpha_f, gamma_i)(b_f,c_i) * PFG()(gamma_i, alpha_i)(c_f,b_i);
                }


            }

        }}

        for (int alpha_f=0; alpha_f<Ns; alpha_f++){
        for (int alpha_i=0; alpha_i<Ns; alpha_i++){

            for (int gamma_f=0; gamma_f<Ns; gamma_f++){
            for (int gamma_i=0; gamma_i<Ns; gamma_i++){

                if (flavor!=2)
                seq_src()(alpha_f, alpha_i)(a_f,a_i) += ee  * 
                    (GFG()(gamma_f,alpha_i)(b_f,b_i) * F()(gamma_f, gamma_i)(c_f,c_i) * P()(gamma_i,alpha_f)()
                    +GFG()(gamma_f,gamma_i)(b_f,b_i) * F()(gamma_f, gamma_i)(c_f,c_i) * P()(alpha_f, alpha_i)());
            }}


        }}

    }}


}