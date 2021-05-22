import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

@st.cache
def load_df(url):
    df = pd.read_csv(url)
    return df


# option
st.set_page_config(page_title="Exoplanet Discovery",
                   page_icon="🪐",
                   layout="wide",
                   initial_sidebar_state="expanded")


#############
## sidebar ##
############# 

st.sidebar.title('Exoplanet Discovery')
st.sidebar.subheader('Navigation')

categorie = st.sidebar.radio("Categories", ("Accueil", "Observer les Exoplanètes",
                                            "Les Exoplanètes habitables",
                                            "L'IA à l'aide des Astrophysicien"))

st.sidebar.title(' ')
option = st.sidebar.beta_expander("Options")
option.markdown(
    """
    L'option _Montre moi la data_ affichera les données 
    qui ont permis de réaliser les graphiques, sous forme de tableaux. 
    """)
show = option.checkbox('Montre moi la data')

expander = st.sidebar.beta_expander("Sources")
expander.markdown(
    """
    __Les bases des données utilisées__ : 
    [NASA Exoplanet Archives](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS) : 
    Data brutes sur les exoplanètes et leur système solaire.
    [Planetary Habitability Laboratory](http://phl.upr.edu/projects/habitable-exoplanets-catalog/data/database) : 
    Détermine quelles sont les exoplanètes habitables ou inhabitables.
    """)

expander.info('Résiliation des **Pirates Ducks** : _Antoine, Franck, Michaël, Mickaël_')
expander.info('Hackathon organisé par la **WildCodeSchool** le 12/05/2021')


##########
## DATA ##
##########

# modifier selon la localisation de la BD
phl_db = 'http://www.hpcf.upr.edu/~abel/phl/hec2/database/phl_exoplanet_catalog.csv'
nea_db = planets.csv

planets = load_df(nea_db)
plan_hab = load_df(phl_db)



###############
## MAIN PAGE ##
###############

if categorie == 'Accueil':
    st.title('Exoplanet Discovery')
    st.subheader('Notre mission : _Donner vie à la data_')

    st.markdown(
        """
        Fermi était septique :
        _« S'il y avait des civilisations extraterrestres, leurs représentants 
        devraient être déjà chez nous. Où sont-ils donc ? »_
        
        Si la question n'a pas de réponse, c'est le principe même de ce paradoxe, 
        elle souligne tout de même la volonté qu'à l'homme de pouvoir rencontrer son alter-égo.
        Si ce n'est pas des civilisations extraterrestres qui nous ont trouvé, alors c'est à nous de les chercher. 
        Les pieds sur terre, la tête dans les étoiles. Nous scrutons le ciel pour 
        trouver une terre qui nous ressemble. Ce sont les _Exoplanètes_.
        """
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.title(" ")
        st.markdown(
            """
            Il faut attendre __1995 pour que la première exoplanète apparaisse__ devant nos yeux et 
            relance la course à la recherche de la vie. Mars n’est plus le seul horizon. 
            L’espoir se propage à présent jusqu’au confins de l’univers.


            C’est aujourd’hui __4383 exoplanètes__ qui ont été découvertes. 
            Dans ce total toutefois, seulement __moins de 1,5% sont considérées remplissant 
            suffisamment de conditions pour accueillir une forme de vie__. 
            """
        )
    with col2:

        temp_tab = planets.groupby((planets['disc_year'] // 10) * 10).count()
        decad_disc = temp_tab[['pl_name']].rename(columns={'pl_name': 'Découvertes'})
        decad_disc['Augmentation'] = ''
        for i in range(1, 5):
            decad_disc.iloc[i, 1] = (((decad_disc.iloc[i, 0] - decad_disc.iloc[i-1, 0]) / decad_disc.iloc[i-1, 0]) * 100).round()

        fig = px.bar(decad_disc, x=decad_disc.index, y="Découvertes", 
                     title="Evolution du nombre d'exoplanètes découvertes",
                     text='Augmentation',
                     color_discrete_sequence=['darkblue']*len(decad_disc))
        fig.update_traces(texttemplate='%{text:.2s}%')
        fig.update_layout(showlegend=True, font_family='IBM Plex Sans', title_x=0.5,
                          xaxis=dict(title="Pourcentage d'évolution d'une décénnie sur l'autre"),
                          yaxis=dict(title="Nombre d'exoplanètes découvertes"),
                          uniformtext_minsize=10, uniformtext_mode='hide',
                          margin=dict(l=40, r=70, b=70, t=70),
                          legend=dict(x=0, y=0.96, traceorder="normal",
                                      bgcolor='rgba(0,0,0,0)',
                                      font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
        """
        Nous vous proposons de partir ensemble pour un voyage dans les méandres de l’univers. 
        Où les températures ardentes flirtent avec le zéro absolu et où le vide est la règle et la vie l’exception.

        Partons ensemble à la rencontre des exoplanetes
        """)

    st.title(" ")
    col1, col2, col3 = st.beta_columns([1, 4, 1])
    with col2:
        st.image("https://i.pinimg.com/originals/b8/d2/bf/b8d2bfc3b9b5cd3224cfe8bda2c928f3.jpg",
                 caption="Ceci n'est pas une exoplanète")

    expander = st.beta_expander("Les technologies utilisées")
    expander.write('Plusieurs librairies de _Python_ ont été utilisées pour la réalisation de ce site : ')
    col1, col2, col3, col4 = expander.beta_columns(4)
    with col1:
        st.write('__Gestion des base de données__')
        st.image('https://datapresta.s3.eu-west-3.amazonaws.com/Assets/tool_pandas.png')
    with col2:
        st.write('__Création du modèle de ML__')
        st.image('data:image/png;base64,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')
    with col3:
        st.write('__Création des graphiques__')
        st.image('data:image/png;base64,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')
    with col4:
        st.write('__Création de la WebApp__')
        st.image('data:image/png;base64,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')

elif categorie == "Observer les Exoplanètes":
    st.title('Comment découvrir des Exoplanètes')
    st.subheader("La découverte d'un nouveau Monde")

    st.markdown(
        """
        Le 6 octobre 1995, les astronomes Michel Mayor et Didier Queloz, annoncent la découverte d'une première 
        exoplanète. Cette planète, nommée __51 Pegasi B__, se  situe à une cinquantaine 
        d'années lumière de la Terre dans la constelation du Pégase.
        """
    )

    fig = px.histogram(planets, x="disc_year", color="discoverymethod",
                       title="<b>Le nombre de planètes découvertes par années et par méthodes</b>",
                       nbins=10, color_discrete_sequence=px.colors.sequential.Agsunset_r,
                       labels="Méthode de découverte")
    fig.update_layout(xaxis_title="Années de découverte", yaxis_title="Nombre d'Exoplanet")
    st.plotly_chart(fig, use_container_width=True)

    if show:
        df_hist = pd.pivot_table(planets, index='disc_year', values='pl_name', columns='discoverymethod',
                                 aggfunc='count', margins=True).fillna(0)
        st.dataframe(df_hist)
    
    st.markdown("""
    ___Qu'est ce que la méthode des vitesses radiales___

    La force de gravité des planètes modifie le déplacement de leur étoile.
    Les capteurs situés sur Terre vont détecter des spectres passant d'une couleur bleu à une couleur rouge. 
    Le décalage de temps durant le changement de couleurs permet de déduire des paramètres 
    physiques comme la vitesse, la masse et la distance.
    
    ___Et la méthode la méthode du transit ?___
    Cette méthode consiste en l'observation d'une répétition constante d'une __variation de luminosité__ d'une étoile.
    Lorsqu'une planète passe devant une étoiles, elle crée une zone d'ombre 
    qui font varier la luminosité captée depuis la Terre.
    """)

    col1, col2, col3 = st.beta_columns([1, 3, 1])
    lk = 'https://raw.githubusercontent.com/MickaelKohler/Exoplanet_Discovery/main/Ressources/Astronomical_Transit.gif'
    with col2:
        st.markdown(f"![Alt Text]({lk})")

    fig = px.scatter(data_frame=planets, x="sy_disterr1", y="pl_orbper",
                     title="<b>Les méthodes utilisées en fonction de la période orbitale et de la distance à la Terre</b>",
                     color='discoverymethod',)
    fig.update_layout(xaxis_title="Distance à la Terre (al)", yaxis_title="Période orbitale autour de l'étoile")
    fig.update_xaxes(range=[-2, 200])
    fig.update_yaxes(range=[0, 200])
    st.plotly_chart(fig, use_container_width=True) 

    st.subheader("La contribution de Kepler dans la recherches d'exoplanètes")
    st.markdown(
        """
        Les méthodes de détection des exoplanètes peuvent être appliquées sur Terre mais aussi directement depuis 
        l'espace. Elles nécessitent l'utilisation d'équipement spécifiques capable d'enregistrer l'image des spectres 
        lumineux. Ces équipements peuvent aller du plus pointus aux simple télescope ou appareil photo.
        """)

    # Groupe les objectifs photos et groupes les telescopes
    planets2 = planets.copy()
    planets2["disc_telescope"] = planets2["disc_telescope"].apply(lambda x: "Objectif photo" if str(x) == 'Canon 400mm f/2.8L' else x)
    planets2["disc_telescope"] = planets2["disc_telescope"].apply(lambda x: "Objectif photo" if str(x) == 'Mamiya 645 80mm f/1.9' else x)
    planets2["disc_telescope"] = planets2["disc_telescope"].apply(lambda x: "Objectif photo" if str(x) == 'Canon 200mm f/1.8L' else x)
    planets2["disc_telescope"] = planets2["disc_telescope"].apply(lambda x: x if (str(x) == '0.95 m Kepler Telescope' or str(x) == 'Objectif photo') else "Telescope")

    fig = px.histogram(planets2, x="disc_telescope", color="discoverymethod",
                       title="<b>Nombre de planètes détectées par type de téléscope</b>"
                       ).update_xaxes(categoryorder="total descending")

    fig.update_layout(xaxis_title="Type de telescope",
                      yaxis_title="Nombre de planètes détéctées",
                      title_text='Max température par Date en fonction des opinions', title_x=0.5)

    col1, col2 = st.beta_columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.title('')
        st.markdown(
            """
            En 2009, l'engin spatial Kepler est envoyé en orbite avec l'objectif de 
            recenser les planètes similaire à la Terre.
            
            Il a été conçu pour utiliser la méthode des transits par l'intermédiaire d télescope de 0.98 mètre 
            de diamètre équipé d'un détecteur mesurant l'intensité lumineuse des étoiles.
            
            La mission de Kepler s'est terminée en 2019 après la découverte record de plus de 2600 exoplanètes. 
            """)


elif categorie == "Les Exoplanètes habitables":
    st.title('Les caractéristiques des Exoplanètes habitables')
    st.subheader('Où sont elles et quels sont leurs projets')
    
    phl_sample = plan_hab[['P_NAME', 'S_TYPE_TEMP', 'P_TYPE', 'S_AGE', 'P_DISTANCE', 'S_TEMPERATURE']]
    zone_hab = pd.merge(planets, phl_sample, left_on='pl_name', right_on='P_NAME', how='left')
    habit = zone_hab[zone_hab['P_HABITABLE'].isin([1, 2])]

    st.markdown(
        """
        On dénombre dans la base de données plus de *** exoplanètes et seulement *** qui sont considérées 
        comme pouvant potentiellement habriter la vie.
        """
    )

    # réparition des planètes
    constelation = planets[planets['P_HABITABLE'].isin([1, 2])][['pl_name', 'hostname', 'S_CONSTELLATION']]
    constelation.dropna(inplace=True)
    fig = px.sunburst(constelation, path=['S_CONSTELLATION', 'hostname', 'pl_name'], maxdepth=2,
                      color_discrete_sequence=px.colors.sequential.Peach_r)
    fig.update_layout(title="<b>Où sont localisées les planètes habitables ?</b>",
                      margin=dict(l=10, r=10, b=10, t=40))

    col1, col2 = st.beta_columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.title(" ")
        st.markdown(
            """
            Le tableau interactif ci-contre vous présente la position de l’ensemble des exoplanètes habitables.
            Vous avez :
            - _Sur le cercle intérieur_ : les constellations.
            - _Sur le cercle extérieur_ : les systèmes solaire.
            
            Vous pouvez cliquer sur le système pour afficher les noms des exoplanètes habitables qui le composent. 
            """)

    planet_name = habit[habit.index == habit['sy_dist'].idxmin()].iloc[0, 0]
    planet_distance = (habit['sy_dist'].min()*3.26156).round(2)
    st.markdown(
        f"""
        __Où se situe la planète la plus proche ?__ La planète potentiellement habitables 
        la plus proche est __{planet_name}__, qui est située à {planet_distance} années lumières.


        A savoir, qu'il faudait _76 624 993 ans_ de voyage à la sonde _Voyager 1_ pour atteindre cette exoplanète.
        
        Pour qu'une planète soit considéré comme habitable, elle doit être située dans la __Zone Habitable__ 
        qui est la région de l’espace où les conditions sont favorables à l’apparition de la vie, 
        telle que nous la connaissons sur Terre.
        Les limites des zones habitables sont calculées à partir des éléments connus de la biosphère de la Terre, 
        comme sa position dans le Système solaire et la quantité d'énergie qu'elle reçoit du Soleil.  
        
        Le graphique ci-dessous permet de bien percevoir cette _Zone Habitable_, 
        les exoplanètes devant s'éloigner à mesure que son étoile gagne en puissance.       
        """
    )

    # zone habitable
    clean_zone = zone_hab[(zone_hab['P_DISTANCE'] < 2) &
                          (zone_hab['S_TEMPERATURE'] > 2500) &
                          (zone_hab['S_TEMPERATURE'] < 8000)]
    clean_zone['P_HABITABLE'] = clean_zone['P_HABITABLE'].apply(lambda x: 'Non Habitable' if x == 0 else 'Habitable')
    inHab = clean_zone[clean_zone['P_HABITABLE'] == 'Non Habitable']
    hab = clean_zone[clean_zone['P_HABITABLE'] == 'Habitable']

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            text=inHab['pl_name'],
            mode='markers',
            x=inHab['P_DISTANCE'],
            y=inHab['S_TEMPERATURE'],
            marker=dict(color='firebrick', opacity=0.3),
            name='Non Habitable'
        )
    )
    fig.add_trace(
        go.Scatter(
            text=hab['pl_name'],
            mode='markers',
            x=hab['P_DISTANCE'],
            y=hab['S_TEMPERATURE'],
            marker=dict(color='lightseagreen'),
            name='Habitable'
        )
    )
    fig.update_layout(
        title='<b>La situation des planètes habitables selon la chaleur du soleil et la distance</b>',
        yaxis=dict(title="Température du soleil (en kelvins)"),
        xaxis=dict(title="Distance planète/étoile (en année-lumière)"),
        margin=dict(l=10, r=10, b=10, t=70))
    st.plotly_chart(fig, use_container_width=True)

    expander = st.beta_expander("Illustration de la zone habitable dans notre système solaire")
    expander.image('https://cdn.shopify.com/s/files/1/0077/5192/5837/files/zone_habitable_systeme_solaire_espace_stellaire_1024x1024.jpg?v=1587994956')

    st.markdown("---")
    
    # Comparatif Habitable/inhabitable
    st.subheader("Qu'est ce qui caractérise une planète habitable ?")
    st.markdown(
        """
        La _Zone Habitable_ met en avant la nécessité de déterminer les critères 
        qui font qu’une exoplanète soit suspectée comme pouvant être habitable. 

       On peut donc tenter de comparer les caractéristiques des exoplanètes 
        considérées comme habitables de l’ensemble des exoplanètes.
        Restons dans les étoiles et essayons de répondre à la question : 
        _Quelle type d’étoile favorise la présence d’exoplanètes habitables ?_
        """
    )

    # Sun Type
    sType = pd.DataFrame(zone_hab['S_TYPE_TEMP'].value_counts(normalize=True)*100).rename(columns={'S_TYPE_TEMP': 'Exoplanètes'})
    sType_hab = habit['S_TYPE_TEMP'].value_counts(normalize=True)*100
    sType_tab = pd.concat([sType, sType_hab], axis=1).reindex(index=['O', 'B', 'A', 'F', 'G', 'K', 'M'])
    sType_tab = sType_tab.fillna(0).rename(columns={'S_TYPE_TEMP': 'Habitables'}).round(2)

    fig = px.bar(sType_tab, x=sType_tab.index, y=["Exoplanètes", "Habitables"], barmode='group',
                 title="<b>La répartition des exoplanètes selon le type de leur étoile</b> (en pourcents)",
                 color_discrete_map={'Exoplanètes': 'darkblue', 'Habitables': 'crimson'})
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=True, font_family='IBM Plex Sans',
                      xaxis=dict(title="Catégorie d'étoile"),
                      yaxis=dict(title=None),
                      uniformtext_minsize=10,
                      uniformtext_mode='hide',
                      margin=dict(l=10, r=10, b=10),
                      legend=dict(x=0, y=1, traceorder="normal", bgcolor='rgba(0,0,0,0)', font=dict(size=12)))
    texts = [sType_tab["Exoplanètes"], sType_tab["Habitables"]]
    for i, t in enumerate(texts):
        fig.data[i].text = t

    if show:
        col1, col2 = st.beta_columns([1, 3])
        with col2:
            st.plotly_chart(fig, use_container_width=True)
        with col1:
            st.title(' ')
            st.dataframe(sType_tab)
    else:
        st.plotly_chart(fig, use_container_width=True)
   
    col1, col2 = st.beta_columns([1, 2])
    with col1:
        st.markdown(
            """
            On peut constater que ce sont surtout les étoiles de type K et M qui comprennent 
            le plus d’exoplanètes habitables. Ce qui s’explique sans doute par 
            le faite que ce sont les plus petites et donc les moins chaudes. 

            Le tableau ci-contre explique la différence entre chaque type.
            """)

    with col2:
        sol_typ = pd.DataFrame(data=[['> 25 000 K', 'bleue', 'azote, carbone, hélium et oxygène'],
                                     ['10 000–25 000 K', 'bleue-blanche', 'hélium, hydrogène'],
                                     ['7 500–10 000 K', 'blanche', 'hydrogène'],
                                     ['6 000–7 500 K', 'jaune-blanche',
                                      'métaux : fer, titane, calcium, strontium et magnésium'],
                                     ['5 000–6 000 K', 'jaune (comme le Soleil)',
                                      'calcium, hélium, hydrogène et métaux'],
                                     ['3 500–5 000 K', 'orange', 'métaux et monoxyde de titane'],
                                     ['< 3 500 K', 'rouge', 'métaux et monoxyde de titane']],
                               index=['O', 'B', 'A', 'F', 'G', 'K', 'M'],
                               columns=['température', 'couleur conventionnelle', "raies d'absorption"])
        st.write(sol_typ)
    
    # Sun Age
    sAge = planets.groupby((zone_hab['S_AGE'] // 2) * 2).count()[['pl_name']]
    sAge.iloc[5, 0] = sAge.iloc[5:, 0].sum()
    sAge['norm'] = ((sAge['pl_name']*100) / sAge['pl_name'].sum()).round(2)
    sAge = sAge.drop(columns=['pl_name']).drop([12, 14]).rename(columns={'norm': 'Exoplanètes'})

    sAge_hab = habit.groupby((habit['S_AGE'] // 2) * 2).count()[['pl_name']]
    sAge_hab['norm'] = ((sAge_hab['pl_name']*100) / sAge_hab['pl_name'].sum()).round(2)
    sAge_hab = sAge_hab.drop(columns=['pl_name']).rename(columns={'norm': 'Habitables'})

    sAge_tab = pd.concat([sAge, sAge_hab], axis=1).fillna(0).round(2)
    sAge_tab.rename(index={0: '<2', 2: '2-4', 4: '4-6', 6: '6-8', 8: '8-10', 10: '+10'}, inplace=True)

    fig = px.bar(sAge_tab, x=sAge_tab.index, y=["Exoplanètes", "Habitables"],
                 title="<b>La répartition des exoplanètes selon l'age de leur étoile</b> (en pourcents)",
                 barmode='group',
                 color_discrete_map={'Exoplanètes': 'darkblue', 'Habitables': 'crimson'})
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=True, font_family='IBM Plex Sans',
                      xaxis=dict(title="Age de l'étoile (Gy)"),
                      yaxis=dict(title=None),
                      uniformtext_minsize=10, uniformtext_mode='hide',
                      margin=dict(l=10, r=10, b=10),
                      legend=dict(x=0, y=1, traceorder="normal",
                                  bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=12)))
    texts = [sAge_tab["Exoplanètes"], sAge_tab["Habitables"]]
    for i, t in enumerate(texts):
        fig.data[i].text = t

    if show:
        col1, col2 = st.beta_columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.title(' ')
            st.dataframe(sAge_tab, height=360)
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        Toujours dans les étoiles, on remarque que les exoplanètes observées sont essentiellement situées 
        sur les __étoiles les plus jeunes__, même si aucune tranche d’âge ne sort du lot. 

        Pour que la vie puisse apparaître sur une planète, il ne suffit pas qu'elle soit dans l'écosphère de son 
        étoile ; son système planétaire doit se situer __assez près du centre de la galaxie__ pour avoir suffisamment 
        d'éléments lourds qui favorisent la formation de planètes telluriques et des 
        atomes nécessaires à la vie (fer, cuivre, etc).
        Mais ce système devra également se situer __assez loin du centre galactique__ pour éviter des dangers tels que 
        le trou noir au centre de la galaxie et les supernova.
        Mais l'exoplanète en elle même doit présenter des conditions intrinsèque pour 
        être une bonne candidate pour accueillir la vie. 
        """
    )

    # Exoplanet type
    pType = pd.DataFrame(zone_hab['P_TYPE'].value_counts(normalize=True)*100).rename(columns={'P_TYPE': 'Exoplanètes'})
    pType_hab = habit['P_TYPE'].value_counts(normalize=True)*100
    pType_tab = pd.concat([pType, pType_hab], axis=1).reindex(index=['Miniterran', 'Subterran', 'Terran',
                                                                     'Superterran', 'Neptunian', 'Jovian'])
    pType_tab = pType_tab.fillna(0).round(2).rename(columns={'P_TYPE': 'Habitables'})

    fig = px.bar(pType_tab,
                 x=pType_tab.index,
                 y=["Exoplanètes", "Habitables"],
                 title="<b>La répartition des exoplanètes selon leur type</b> (en pourcents)",
                 barmode='group',
                 color_discrete_map={'Exoplanètes': 'darkblue', 'Habitables': 'crimson'})
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=True, font_family='IBM Plex Sans',
                      xaxis=dict(title="Type d'exoplanète"),
                      yaxis=dict(title=None),
                      uniformtext_minsize=10, uniformtext_mode='hide',
                      margin=dict(l=10, r=10, b=10),
                      legend=dict(x=0,
                                  y=1,
                                  traceorder="normal",
                                  bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=12)))
    texts = [pType_tab["Exoplanètes"], pType_tab["Habitables"]]
    for i, t in enumerate(texts):
        fig.data[i].text = t

    col1, col2 = st.beta_columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if show:
            st.title(" ")
            st.dataframe(pType_tab)
        else:
            st.title(" ")
            st.markdown(
                """
                Les type d'exoplanet selon
                la masse de la terre (MT): 
                - _Miniterran_ : -0,1 MT
                - _Subterran_ : 0,1 à 0,5 MT
                - _Terran_ : 0,5 à 2 MT
                - _Superterran_ : 2 à 10 MT
                - _Neptunian_ : 10 à 50 MT
                - _Jovian_ : +50 MT 
                """
            )
    
    st.markdown(
        """
        Les exoplanète habitables sont essentiellement situées sur des planètes équivalentes 
        à la terre ou légèrement plus grosse. Comme pour la _Zone Habitable_, la conditions de 
        validité pour être considérée comme une exoplanète habitable est très restreinte. 
        Ces conditions ne sont bien sur pas limitatives. Il existe de nombreux critères à prendre en compte. 
        De nombreuses variables qui peuvent être étudiées par un algorithme afin de 
        pouvoir créer un modèle permettant de repérer les exoplanètes.
        """
    )


elif categorie == "L'IA à l'aide des Astrophysicien":
    st.title("L'intelligence artificielle à la recherche de la vie")
    st.subheader("Comment le Machine Learning peut venir à l'aide des Astrophysicien")
    st.title(" ")

    st.markdown(
        """
        Lors de notre recherche de la base de donnée parfaite (BDP), nous avons trouvé une base de donnée 
        hébergée par _Planetary Hability Laboratory_ qui tente de répertorier et identifier les exoplanètes habitables.

        Toutefois, leur base de donnée ne prend pas en considération les dernières 
        exoplanètes découvertes à partir de début 2020. 
        Nous avons donc tenté d’entrainer un __algorithme de Machine Learning__ pour déterminer, 
        selon les caractéristiques de chaque exoplanète, si elle peut être catégorisée comme habitable ou non, 
        dans le but de catégoriser celles qui n’ont pas été identifiée.
        """
    )

    df_exoplanet_vf = planets.copy()

    # Selecting all numerical column from dataframe
    numeical_columns_list = df_exoplanet_vf.select_dtypes(include=np.number).columns.tolist()
    df_exoplanet_num = df_exoplanet_vf[numeical_columns_list]

    # Selecting main categorical columns
    df_exoplanet_cat = df_exoplanet_vf[['pl_letter', 'discoverymethod', 'disc_locale']]

    # setting them into numerical value using factorization
    df_exoplanet_cat['pl_letter'] = df_exoplanet_cat['pl_letter'].factorize()[0]
    df_exoplanet_cat['discoverymethod'] = df_exoplanet_cat['discoverymethod'].factorize()[0]
    df_exoplanet_cat['disc_locale'] = df_exoplanet_cat['disc_locale'].factorize()[0]

    # merging dataset of selected columns 
    df_exoplanet_rf = df_exoplanet_num.join(df_exoplanet_cat)

    # ...and splitting dataset on 'P_HABITABLE' none or not
    df_exoplanet_rf_1 = df_exoplanet_rf[df_exoplanet_rf['P_HABITABLE'].notna()]
    df_exoplanet_rf_2 = df_exoplanet_rf[df_exoplanet_rf['P_HABITABLE'].isna()]

    # filling missing values with the mean of each column
    df_exoplanet_rf_1.fillna(df_exoplanet_rf_1.mean(), inplace=True)

    # starting ML with XGboost
    y = df_exoplanet_rf_1["P_HABITABLE"]
    X = df_exoplanet_rf_1.drop("P_HABITABLE", axis=1)

    # training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

    # fitting model on training data
    model = XGBClassifier().fit(X, y)

    # making prediction on unknown dataset
    df_exoplanet_rf_2["predictions"] = model.predict(df_exoplanet_rf_2.drop(columns='P_HABITABLE'))

    df_test = df_exoplanet_vf[['pl_name', 'discoverymethod']]
    df_final = pd.merge(df_test, df_exoplanet_rf_2, left_index=True, right_index=True)
    df_final = df_final[['pl_name', 'disc_year', 'discoverymethod_x', 'predictions']]
    df_final.loc[df_final['predictions'] == 0, 'predictions'] = 'Inhabitable'

    df_final.rename(columns={'pl_name': "Nom de l'Exoplanète",
                             'discoverymethod_x': 'Méthode utilisée',
                             'disc_year': 'Découverte',
                             'predictions': 'Prédiction'}, inplace=True)
    
    st.title(' ')
    ML_off = True
    col1, col2 = st.beta_columns([1, 3])
    with col1:
        st.markdown(
            """
            __Cliquez sur le bouton ci-dessous__ pour rechercher de nouvelles planètes 
            ayant le potentielle d'être habitable. 
            """
        )
        if st.button('Rechercher la vie'):
            ML_off = False
            st.markdown(
                """
                Comme vous pouvez le voir, __aucune nouvelle exoplanète ne 
                remplit les conditions__ pour pouvoir accueillir la vie. 

                La recherche continue…
                _« I want to believe »_
                """
            )
    with col2:
        if ML_off:
            display_tab = df_final.copy()
            display_tab['Prédiction'] = ' '
            st.dataframe(display_tab, height=550)
        else:
            st.dataframe(df_final, height=550)

    expander = st.beta_expander("Explication du modèle retenu")
    expander.markdown(
        """
        ___Quel modèle a été retenu ?___
        Nous avons testé les algorithmes de classification les plus 
        pertinents afin de prédire si une planète est habitable.
        Lors de ces tests, les algorithmes, ci-dessous, ont produit les résultats les plus proches 
        de la réalités (scores), c'est à dire en comparant nos résultats aux informations à notre disposition.
        Bien que les meilleurs scores soient supérieurs à celui du XGBoost, que nous avons choisit, 
        ce dernier a été plus à même de prédire les planètes habitables connues.
        """)
        
    # New dataframe with score of different test
    dataScore = {'Test': ['SGDClassifier', "DecisionTreeClassifier", "KNeighborsClassifier", "BaggingClassifier",
                          "RandomForestClassifier", "AdaBoostClassifier", "XGBoost"],
                 "Score": [0.990069513406156, 0.984111221449851, 0.991062562065541, 0.990069513406156,
                           0.991062562065541, 0.985104270109235, 0.9890764647467726]}
    pd.DataFrame.from_dict(dataScore)

    fig = px.histogram(data_frame=dataScore,
                       x="Test",
                       y="Score",
                       color_discrete_sequence=['darkblue'] ,
                       title="Score des différents test").update_xaxes(categoryorder="total descending")

    fig.update_yaxes(range=[0.97, 1])
    fig.update_layout(xaxis_title="Score", yaxis_title="Test")

    expander.plotly_chart(fig, use_container_width=True) 

    expander.markdown(
        """
        ___Qu'est-ce que le XGBoost ?___
        XGBoot est la Extrême Gradient Boosted Trees, plus simplement 
        il s'agit d'une forêt d'arbres de décision optimisée.
        "Un arbre de décision est un outil d'aide à la décision représentant 
        un ensemble de choix sous la forme graphique d'un arbre. 
        Les différentes décisions possibles sont situées aux extrémités des branches (les « feuilles » de l'arbre), 
        et sont atteintes en fonction de décisions prises à chaque étape" 
        [source](https://fr.wikipedia.org/wiki/Arbre_de_d%C3%A9cision)
        """
    )
