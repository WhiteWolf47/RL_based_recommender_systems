import pandas as pd
movie_df = pd.read_csv('wiki_movie_plots_deduped.csv')
def Data_Cleaning(Genre):
    '''
    Here we have cleaned the entire Genre column of the dataset by removing unwanted symbols, categories, and 
    replacing categories which meant the same with a common category name. It reduduces our number of target labels.
    NOTE: This function is inspired from the kernel - https://www.kaggle.com/aminejallouli/genre-classification-based-on-wiki-movies-plots
    I have only improved it a bit further according to my requirements.
    '''
    movie_df['Genre_improved'] = movie_df['Genre']
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.strip()
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' - ', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' / ', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('/', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' & ', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(', ', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('; ', '|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('bio-pic', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biopic', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biographical', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biodrama', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('bio-drama', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biographic', 'biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(film genre\)', '')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('animated','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('anime','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('children\'s','children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedey','comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\[not in citation given\]','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' set 4,000 years ago in the canadian arctic','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('historical','history')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romantic','romance')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('3-d','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('3d','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('viacom 18 motion pictures','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sci-fi','science_fiction')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('ttriller','thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('.','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('based on radio serial','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' on the early years of hitler','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sci fi','science_fiction')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('science fiction','science_fiction')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' (30min)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('16 mm film','short')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\[140\]','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\[144\]','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' for ','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('adventures','adventure')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('kung fu','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('kung-fu','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('martial arts','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('world war ii','war')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('world war i','war')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biography about montreal canadiens star|maurice richard','biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('bholenath movies|cinekorn entertainment','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(volleyball\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('spy film','spy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('anthology film','anthology')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biography fim','biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('avant-garde','avant_garde')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biker film','biker')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('buddy cop','buddy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('buddy film','buddy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedy 2-reeler','comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('films','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('film','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biography of pioneering american photographer eadweard muybridge','biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('british-german co-production','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('bruceploitation','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedy-drama adaptation of the mordecai richler novel','comedy-drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('movies by the mob\|knkspl','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('movies','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('movie','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('coming of age','coming_of_age')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('coming-of-age','coming_of_age')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('drama about child soldiers','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('(( based).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('(( co-produced).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('(( adapted).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('(( about).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('musical b','musical')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('animationchildren','animation|children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' period','period')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('drama loosely','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(aquatics|swimming\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(aquatics|swimming\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace("yogesh dattatraya gosavi's directorial debut \[9\]",'')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace("war-time","war")
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace("wartime","war")
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace("ww1","war")
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('unknown','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace("wwii","war")
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('psychological','psycho')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('rom-coms','romance')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('true crime','crime')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|007','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('slice of life','slice_of_life')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('computer animation','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('gun fu','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('j-horror','horror')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(shogi|chess\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('afghan war drama','war drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|6 separate stories','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(30min\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' (road bicycle racing)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' v-cinema','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('tv miniseries','tv_miniseries')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|docudrama','\|documentary|drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' in animation','|animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('((adaptation).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('((adaptated).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('((adapted).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('(( on ).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('american football','sports')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dev\|nusrat jahan','sports')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('television miniseries','tv_miniseries')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(artistic\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \|direct-to-dvd','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('history dram','history drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('martial art','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('psycho thriller,','psycho thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|1 girl\|3 suitors','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' \(road bicycle racing\)','')
    filterE = movie_df['Genre_improved']=="ero"
    movie_df.loc[filterE,'Genre_improved']="adult"
    filterE = movie_df['Genre_improved']=="music"
    movie_df.loc[filterE,'Genre_improved']="musical"
    filterE = movie_df['Genre_improved']=="-"
    movie_df.loc[filterE,'Genre_improved']=''
    filterE = movie_df['Genre_improved']=="comedy–drama"
    movie_df.loc[filterE,'Genre_improved'] = "comedy|drama"
    filterE = movie_df['Genre_improved']=="comedy–horror"
    movie_df.loc[filterE,'Genre_improved'] = "comedy|horror"
    
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(' ','|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace(',','|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('-','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionadventure','action|adventure')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actioncomedy','action|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actiondrama','action|drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionlove','action|love')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionmasala','action|masala')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionchildren','action|children')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('fantasychildren\|','fantasy|children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('fantasycomedy','fantasy|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('fantasyperiod','fantasy|period')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('cbctv_miniseries','tv_miniseries')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramacomedy','drama|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramacomedysocial','drama|comedy|social')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramathriller','drama|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedydrama','comedy|drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramathriller','drama|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedyhorror','comedy|horror')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sciencefiction','science_fiction')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('adventurecomedy','adventure|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('animationdrama','animation|drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|\|','|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('muslim','religious')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('thriler','thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('crimethriller','crime|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('fantay','fantasy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionthriller','action|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedysocial','comedy|social')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('martialarts','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|\(children\|poker\|karuta\)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('epichistory','epic|history')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('erotica','adult')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('erotic','adult')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('((\|produced\|).+)','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('chanbara','chambara')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('comedythriller','comedy|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biblical','religious')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biblical','religious')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('colour\|yellow\|productions\|eros\|international','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|directtodvd','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('liveaction','live|action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('melodrama','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('superheroes','superheroe')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('gangsterthriller','gangster|thriller')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('heistcomedy','comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('heist','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('historic','history')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('historydisaster','history|disaster')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('warcomedy','war|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('westerncomedy','western|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('ancientcostume','costume')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('computeranimation','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramatic','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('familya','family')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('familya','family')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramedy','drama|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dramaa','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('famil\|','family')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('superheroe','superhero')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('biogtaphy','biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('devotionalbiography','devotional|biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('docufiction','documentary|fiction')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('familydrama','family|drama')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('espionage','spy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('supeheroes','superhero')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romancefiction','romance|fiction')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('horrorthriller','horror|thriller')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('suspensethriller','suspense|thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('musicaliography','musical|biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('triller','thriller')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|\(fiction\)','|fiction')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romanceaction','romance|action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romancecomedy','romance|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romancehorror','romance|horror')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romcom','romance|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('rom\|com','romance|comedy')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('satirical','satire')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('science_fictionchildren','science_fiction|children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('homosexual','adult')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sexual','adult')

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('mockumentary','documentary')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('periodic','period')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('romanctic','romance')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('politics','political')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('samurai','martial_arts')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('tv_miniseries','series')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('serial','series')

    filterE = movie_df['Genre_improved']=="musical–comedy"
    movie_df.loc[filterE,'Genre_improved'] = "musical|comedy"

    filterE = movie_df['Genre_improved']=="roman|porno"
    movie_df.loc[filterE,'Genre_improved'] = "adult"


    filterE = movie_df['Genre_improved']=="action—masala"
    movie_df.loc[filterE,'Genre_improved'] = "action|masala"


    filterE = movie_df['Genre_improved']=="horror–thriller"
    movie_df.loc[filterE,'Genre_improved'] = "horror|thriller"

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('family','children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('martial_arts','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('horror','thriller')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('war','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('adventure','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('science_fiction','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('western','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('western','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('noir','black')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('spy','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('superhero','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('social','')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('suspense','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sex','adult')


    filterE = movie_df['Genre_improved']=="drama|romance|adult|children"
    movie_df.loc[filterE,'Genre_improved'] = "drama|romance|adult"

    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('\|–\|','|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.strip(to_strip='\|')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('actionner','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('love','romance')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('crime','mystery')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('kids','children')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('boxing','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('buddy','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('cartoon','animation')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('cinema','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('religious','supernatural')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('christian','supernatural')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('lgbtthemed','romance')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('detective','mystery')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('nature','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('fiction','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('music','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('musical','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('short','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('mythology','supernatural')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('mythological','supernatural')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('masala','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('military','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('sexploitation','adult')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('tragedy','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('murder','mystery')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('disaster','drama')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('documentary','biography')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('dance','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('cowboy','action')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('anthology','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('artistical','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.replace('art','artistic')
    movie_df['Genre_improved']=movie_df['Genre_improved'].str.strip()
    return movie_df['Genre_improved']