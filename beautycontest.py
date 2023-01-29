def data():
    f=input("Chemin du fichier")
    import pandas as pd
    df=pd.read_excel(f) #Ouvrir les donnees et les mettres en dataframe#
    df= df.rename(columns={'v01_case':'ident','v02_treatment':'Grpi','v03_experiment':'exper','v041_choice':'Val1','v042_choice':'Val2','v043_choice':'Val3','v044_choice':'Val4'})#Transformation des noms des variables#
    df= df.fillna(0.0)#Mettre des 0 a la place des valeurs manquante pour pourvoir faire l'addition des valeurs#
    df['val']= df['Val1'] + df['Val2'] +df['Val3'] + df['Val4']#Nouvelle colonne avec l'integralite des valeurs choisis#
    df= df.drop(['Val1', 'Val2', 'Val3', 'Val4'], axis=1)
    return df
#data()

def moyenneexp(df,m): #Création d'une data frame avec toutes les moyennes et ecartypes par expérience#
    import pandas as pd
    for n in range(1,18,1):       
        l = []
        for i in range(0,len(df['exper']),1):
            if df['exper'][i] == n: 
                l.append(df['ident'][i])
        m.loc[n,'Means'] = df['val'][l[0]:l[-1]].mean() 
        m.loc[n,'std'] = df['val'][l[0]:l[-1]].std()
    return
#moyenneexp(df,m)

def moyennegroup(df,M): #Création d'une data frame avec toutes les moyennes et ecartypes par groupe#
    import pandas as pd
    for n in range(1,12,1):       
        l = []
        for i in range(0,len(df['exper']),1):
           if df['Grpi'][i] == n: 
               l.append(df['ident'][i])  
        M.loc[n,'Means'] = df['val'][l[0]:l[-1]].mean() 
        M.loc[n,'std'] = df['val'][l[0]:l[-1]].std()
    return
#moyennegroup(df,M)

def ajoute(M): #On complete les dataframes avec des variables 'explicatives des moyennes'#
    M['nbpart']=[67, 19, 138, 119, 54, 59, 33, 150, 1458, 3696, 2728] #Nombre de participants# 
    M['timeup']=[5,5,5,10080,30240,5,10080,0,20160,10080,20160] #Temps de reponse accorde en minutes#
    M['payoffs']=[12,20,24,24,18,20,0,18,950,800,600] #Recompenses accordé en dollard#
    M['Knowledge']=[0,0,1,1,2,2,2,0,0,0,0] #Connaissances en théorie des jeux : 0 aucunes, 1 un peu, 2 experts#
    M['nbexper']=[4,1,2,2,1,2,1,1,1,1,1] #Nombre d'experiences faites#
    return 
#ajoute(M)

def reg(M): #Regression des moyennes#
    import pandas as pd
    import statsmodels.api as sm
    X=M[['nbpart','timeup','payoffs','Knowledge','nbexper']]
    Y=M['Means']
    X = sm.add_constant(X)
    model = sm.OLS(Y.astype(float), X.astype(float)).fit()
    global predictions
    predictions = model.predict(X)
    global print_model
    print_model = model.summary()
    return print_model
#reg(M)

def statexp(df): #obtenir les stat d'une expérience#
    n=int(input("Écrire le numero d'experience à analyser"))
    l=[]
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == n:
            l.append(df['ident'][i])
    print(df['val'][l[0]:l[-1]].describe())
    return
#statexp(df)

def statgrp(df): #obtenir les stat d'un groupe#
    N=int(input("Écrire le numero du groupe à analyser"))
    l=[]
    for i in range(0,len(df['Grpi']),1):
        if df['Grpi'][i] == N:
            l.append(df['ident'][i])
    print(df['val'][l[0]:l[-1]].describe())
    return
#statgrp(df)

def Graphgroup(df): #histo par groupes#
    N=int(input("Écrire le numero du groupe à representer graphiquement"))
    import matplotlib.pyplot as plt 
    l = []
    for i in range(0,len(df['Grpi']),1):
        if df['Grpi'][i] == N:
            l.append(df['ident'][i])
    plt.hist(df['val'][l[0]:l[-1]], color='green')
    plt.title("Histogramme des jeux du groupe")
    plt.xlabel("numéro")
    plt.ylabel("frequence")
    plt.xlim(-10,100)
    plt.show
    return
#Graphgroup(df)

def Graphexp(df): #histo par experiences#
    n=int(input("Écrire le numero d'experience à representer graphiquement"))
    import matplotlib.pyplot as plt 
    l = []
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == n:
            l.append(df['ident'][i])
    plt.hist(df['val'][l[0]:l[-1]], color='blue')
    plt.title("Histograme des jeux du groupe")
    plt.xlabel("numéro")
    plt.ylabel("frequence")
    plt.xlim(-10,100)
    plt.show
#Graphexp(df)

def REGraph(M): #Nuage de point avec ols simple et regression simple#
    i=input("Écrire le nom de la variable à regresser")
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np
    Y=M['Means']
    X=M[['nbpart','timeup','payoffs','Knowledge','nbexper']]
    sns.regplot(x=X[i].astype(float),y=Y.astype(float),fit_reg=True)
    plt.show()
    models = sm.OLS(Y.astype(float), X[i].astype(float)).fit()
    print(models.summary())
#REGraph(M)

def testM(df): #Test de diffèrence de moyennes entre 2 expériences (o et u) = affiche la stat de test et la p-value#
    o=int(input("Écrire le numero de la première experience à analyser"))
    u=int(input("Écrire le numero de la deuxième experience à analyser"))
    import scipy.stats as stat
    l = []
    k = []
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == o:
            l.append(df['ident'][i])
    for t in range(0,len(df['exper']),1):
        if df['exper'][t] == u:
            k.append(df['ident'][t])
    print(stat.ttest_ind(df['val'][l[0]:l[-1]],df['val'][k[0]:k[-1]],equal_var=False))
#testM(df)


##### Fonctions modifiées pour faciliter l'analyse#####
def graphexp(df,n): #histo par experiences#
    import matplotlib.pyplot as plt 
    l = []
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == n:
            l.append(df['ident'][i])
    plt.hist(df['val'][l[0]:l[-1]], color='blue')
    plt.title("Histograme des jeux du groupe")
    plt.xlabel("numéro")
    plt.ylabel("frequence")
    plt.xlim(-10,100)
    plt.show
#graphexp(df,n)

def Statexp(df,n): #obtenir les stat d'une expérience#
    l=[]
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == n:
            l.append(df['ident'][i])
    print(df['val'][l[0]:l[-1]].describe())
    return
#Statexp(df,n)

def TestM(df,o,u): #Test de diffèrence de moyennes entre 2 expériences (o et u) = affiche la stat de test et la p-value#
    import scipy.stats as stat
    l = []
    k = []
    for i in range(0,len(df['exper']),1):
        if df['exper'][i] == o:
            l.append(df['ident'][i])
    for t in range(0,len(df['exper']),1):
        if df['exper'][t] == u:
            k.append(df['ident'][t])
    print(stat.ttest_ind(df['val'][l[0]:l[-1]],df['val'][k[0]:k[-1]],equal_var=False))
#TestM(df,o,u)

def ReGraph(M,i): #Nuage de point avec ols simple et regression simple#
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np
    Y=M['Means']
    X=M[['nbpart','timeup','payoffs','Knowledge','nbexper']]
    sns.regplot(x=X[i].astype(float),y=Y.astype(float),fit_reg=True)
    plt.show()
    models = sm.OLS(Y.astype(float), X[i].astype(float)).fit()
    print(models.summary())
#ReGraph(M,i)

def graphgroup(df,N): #histo par groupes#
    import matplotlib.pyplot as plt 
    l = []
    for i in range(0,len(df['Grpi']),1):
        if df['Grpi'][i] == N:
            l.append(df['ident'][i])
    plt.hist(df['val'][l[0]:l[-1]], color='green')
    plt.title("Histogramme des jeux du groupe")
    plt.xlabel("numéro")
    plt.ylabel("frequence")
    plt.xlim(-10,100)
    plt.show
    return
#graphgroup(df,N)

def Statgrp(df,N): #obtenir les stat d'un groupe#
    l=[]
    for i in range(0,len(df['Grpi']),1):
        if df['Grpi'][i] == N:
            l.append(df['ident'][i])
    print(df['val'][l[0]:l[-1]].describe())
    return
#Statgrp(df,N)